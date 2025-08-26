import os, re, json, datetime, logging, hashlib
from pathlib import Path
import pandas as pd
import pdfplumber, camelot, pytesseract
from pdf2image import convert_from_path
from unidecode import unidecode
import chardet
from typing import List, Dict, Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
import time
from functools import partial

warnings.filterwarnings('ignore')

# ====== CONFIG ======
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "kb"
LOG_DIR = ROOT / "logs"
BACKUP_DIR = ROOT / "backups"

# Create directories
for dir_path in [LOG_DIR, OUT_DIR, BACKUP_DIR]:
    dir_path.mkdir(exist_ok=True)

MASTER_CSV = OUT_DIR / "master_data.csv"
MASTER_PARQUET = OUT_DIR / "master_data.parquet"
META_JSON = OUT_DIR / "metadata.json"
SCHEMA_JSON = OUT_DIR / "schema.json"

# Parallel processing configuration
MAX_WORKERS = min(cpu_count(), 8)  # Adjust based on your system
CHUNK_SIZE = 10  # Files per batch
PDF_TIMEOUT = 300  # 5 minutes per PDF
MEMORY_LIMIT_MB = 2048  # Memory limit per process

# ====== THREAD-SAFE LOGGING ======
log_file = LOG_DIR / f"master_builder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MasterBuilder")

# Thread-safe progress tracking
class ProgressTracker:
    def __init__(self, total_files):
        self.total_files = total_files
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, success=True):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            
            # Progress reporting
            progress = (self.completed / self.total_files) * 100
            elapsed = time.time() - self.start_time
            eta = (elapsed / self.completed * (self.total_files - self.completed)) if self.completed > 0 else 0
            
            logger.info(f"Progress: {self.completed}/{self.total_files} ({progress:.1f}%) | "
                       f"Success: {self.successful} | Failed: {self.failed} | "
                       f"ETA: {eta/60:.1f}m")

# ====== ENHANCED HELPERS ======
def detect_encoding(file_path: Path) -> str:
    """Detect file encoding for better CSV reading"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
    except:
        return 'utf-8'

def slug(s: str) -> str:
    """Create URL-safe column names while preserving meaning"""
    if not s or pd.isna(s): return "unknown_col"
    s = str(s).strip()
    
    # Preserve common symbols in M365 context
    s = s.replace('>', '_gt_').replace('<', '_lt_').replace('&', '_and_')
    s = s.replace('%', '_pct_').replace('#', '_num_').replace('$', '_dollar_')
    s = s.replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_')
    
    # Convert to ASCII and lowercase
    s = unidecode(s).lower()
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    
    return s[:100] if s else "unknown_col"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names with collision detection"""
    if df.empty or len(df.columns) == 0:
        return df
    
    original_cols = list(df.columns)
    new_cols = []
    col_counter = {}
    
    for col in original_cols:
        base_name = slug(col)
        
        if base_name in col_counter:
            col_counter[base_name] += 1
            final_name = f"{base_name}_{col_counter[base_name]}"
        else:
            col_counter[base_name] = 0
            final_name = base_name
        
        new_cols.append(final_name)
    
    df.columns = new_cols
    df.attrs['original_columns'] = dict(zip(new_cols, original_cols))
    return df

def safe_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert data types while preserving information"""
    if df.empty:
        return df
    
    original_dtypes = df.dtypes.to_dict()
    
    for col in df.columns:
        try:
            df[col] = df[col].replace(['', 'NULL', 'null', 'None', 'N/A', 'n/a'], pd.NA)
            df[col] = df[col].astype('string')
        except Exception as e:
            logger.debug(f"Type conversion failed for column {col}: {e}")
            df[col] = df[col].astype(str)
    
    df.attrs['original_dtypes'] = original_dtypes
    return df

def create_file_hash(file_path: Path) -> str:
    """Create MD5 hash of file for change detection"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return ""

def clean_dataframe(df: pd.DataFrame, source_info: dict) -> pd.DataFrame:
    """Enhanced DataFrame cleaning with source tracking"""
    if df.empty:
        return df
    
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    if df.empty:
        return df
    
    for key, value in source_info.items():
        if key not in df.columns:
            df[key] = value
    
    df['processed_at'] = datetime.datetime.now().isoformat()
    df['row_id'] = range(len(df))
    
    return df

# ====== PARALLEL EXTRACTORS ======
def process_single_csv(file_path: Path) -> Tuple[str, List[pd.DataFrame], Dict]:
    """Process a single CSV file - optimized for parallel execution"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Processing CSV: {file_path.name}")
    
    frames = []
    stats = {'file': file_path.name, 'type': 'csv', 'status': 'failed'}
    
    try:
        encoding = detect_encoding(file_path)
        
        strategies = [
            {'encoding': encoding, 'sep': ','},
            {'encoding': 'utf-8', 'sep': ','},
            {'encoding': 'latin-1', 'sep': ','},
            {'encoding': encoding, 'sep': ';'},
            {'encoding': encoding, 'sep': '\t'},
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                df = pd.read_csv(file_path, **strategy)
                
                if not df.empty and len(df.columns) > 1:
                    df = normalize_columns(df)
                    df = safe_convert_types(df)
                    
                    source_info = {
                        'source_file': file_path.name,
                        'source_type': 'csv',
                        'extraction_strategy': f"csv_strategy_{i+1}",
                        'file_size': file_path.stat().st_size,
                        'file_hash': create_file_hash(file_path),
                        'thread_name': thread_name
                    }
                    
                    df = clean_dataframe(df, source_info)
                    frames.append(df)
                    
                    stats.update({
                        'status': 'success',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'strategy': i+1
                    })
                    
                    logger.info(f"[{thread_name}] ✅ CSV {file_path.name}: {len(df)} rows")
                    break
                    
            except Exception as e:
                logger.debug(f"[{thread_name}] CSV strategy {i+1} failed: {e}")
                continue
        
        if not frames:
            stats['error'] = 'No valid data extracted'
            
    except Exception as e:
        stats['error'] = str(e)
        logger.error(f"[{thread_name}] ❌ CSV error {file_path.name}: {e}")
    
    return file_path.name, frames, stats

def process_single_excel(file_path: Path) -> Tuple[str, List[pd.DataFrame], Dict]:
    """Process a single Excel file - optimized for parallel execution"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Processing Excel: {file_path.name}")
    
    frames = []
    stats = {'file': file_path.name, 'type': 'excel', 'status': 'failed', 'sheets': []}
    
    try:
        xls = pd.ExcelFile(file_path)
        
        for sheet_name in xls.sheet_names:
            try:
                for header_row in [0, 1, 2, None]:
                    try:
                        df = pd.read_excel(
                            file_path, 
                            sheet_name=sheet_name, 
                            header=header_row,
                            dtype=str,
                            na_filter=False
                        )
                        
                        if not df.empty and len(df.columns) > 1:
                            if header_row is not None:
                                first_row_nulls = df.iloc[0].isna().sum() if len(df) > 0 else len(df.columns)
                                if first_row_nulls > len(df.columns) * 0.7:
                                    continue
                            
                            df = normalize_columns(df)
                            df = safe_convert_types(df)
                            
                            source_info = {
                                'source_file': file_path.name,
                                'source_type': 'excel',
                                'sheet_name': sheet_name,
                                'header_row': header_row,
                                'file_size': file_path.stat().st_size,
                                'file_hash': create_file_hash(file_path),
                                'thread_name': thread_name
                            }
                            
                            df = clean_dataframe(df, source_info)
                            frames.append(df)
                            
                            stats['sheets'].append({
                                'sheet': sheet_name,
                                'rows': len(df),
                                'columns': len(df.columns)
                            })
                            
                            logger.info(f"[{thread_name}] ✅ Excel {file_path.name}[{sheet_name}]: {len(df)} rows")
                            break
                            
                    except Exception:
                        continue
                        
            except Exception as e:
                logger.debug(f"[{thread_name}] Sheet {sheet_name} failed: {e}")
        
        if frames:
            stats['status'] = 'success'
            stats['total_rows'] = sum(len(df) for df in frames)
        else:
            stats['error'] = 'No sheets processed successfully'
            
    except Exception as e:
        stats['error'] = str(e)
        logger.error(f"[{thread_name}] ❌ Excel error {file_path.name}: {e}")
    
    return file_path.name, frames, stats

def process_single_pdf(file_path: Path) -> Tuple[str, List[pd.DataFrame], Dict]:
    """Process a single PDF file - optimized for parallel execution with timeout"""
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Processing PDF: {file_path.name}")
    
    frames = []
    stats = {'file': file_path.name, 'type': 'pdf', 'status': 'failed', 'methods': []}
    
    try:
        start_time = time.time()
        
        # Method 1: PDFplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    if time.time() - start_time > PDF_TIMEOUT:
                        raise TimeoutError("PDF processing timeout")
                    
                    try:
                        tables = page.extract_tables()
                        if not tables:
                            tables = page.extract_tables(table_settings={
                                "vertical_strategy": "lines_strict",
                                "horizontal_strategy": "lines_strict"
                            })
                        
                        for table_num, table in enumerate(tables or [], 1):
                            if table and len(table) > 1:
                                headers = table[0] if table[0] else [f"col_{i}" for i in range(len(table[0]))]
                                data = table[1:] if len(table) > 1 else []
                                
                                if data:
                                    df = pd.DataFrame(data, columns=headers)
                                    df = normalize_columns(df)
                                    df = safe_convert_types(df)
                                    
                                    source_info = {
                                        'source_file': file_path.name,
                                        'source_type': 'pdf',
                                        'extraction_method': 'pdfplumber',
                                        'page_number': page_num,
                                        'table_number': table_num,
                                        'file_size': file_path.stat().st_size,
                                        'file_hash': create_file_hash(file_path),
                                        'thread_name': thread_name
                                    }
                                    
                                    df = clean_dataframe(df, source_info)
                                    frames.append(df)
                    
                    except Exception as e:
                        logger.debug(f"[{thread_name}] PDFplumber page {page_num} failed: {e}")
            
            if frames:
                stats['methods'].append({'method': 'pdfplumber', 'tables': len(frames)})
                
        except Exception as e:
            logger.debug(f"[{thread_name}] PDFplumber failed: {e}")
        
        # Method 2: Camelot (if PDFplumber didn't work or got few results)
        if len(frames) < 2 and time.time() - start_time < PDF_TIMEOUT:
            try:
                tables = camelot.read_pdf(str(file_path), pages='all', flavor='lattice')
                if not tables:
                    tables = camelot.read_pdf(str(file_path), pages='all', flavor='stream')
                
                camelot_frames = 0
                for i, table in enumerate(tables, 1):
                    if time.time() - start_time > PDF_TIMEOUT:
                        break
                    
                    if not table.df.empty:
                        df = normalize_columns(table.df.copy())
                        df = safe_convert_types(df)
                        
                        source_info = {
                            'source_file': file_path.name,
                            'source_type': 'pdf',
                            'extraction_method': 'camelot',
                            'table_number': i,
                            'accuracy': getattr(table, 'accuracy', 0),
                            'file_size': file_path.stat().st_size,
                            'file_hash': create_file_hash(file_path),
                            'thread_name': thread_name
                        }
                        
                        df = clean_dataframe(df, source_info)
                        frames.append(df)
                        camelot_frames += 1
                
                if camelot_frames > 0:
                    stats['methods'].append({'method': 'camelot', 'tables': camelot_frames})
                    
            except Exception as e:
                logger.debug(f"[{thread_name}] Camelot failed: {e}")
        
        # Method 3: OCR fallback (only if no tables found)
        if not frames and time.time() - start_time < PDF_TIMEOUT:
            try:
                pages = convert_from_path(file_path, dpi=150)  # Lower DPI for speed
                
                all_text = []
                for page_num, page_image in enumerate(pages[:5], 1):  # Limit to first 5 pages
                    if time.time() - start_time > PDF_TIMEOUT:
                        break
                    
                    try:
                        text = pytesseract.image_to_string(page_image, config='--psm 6')
                        if text.strip():
                            all_text.append(f"=== Page {page_num} ===\n{text}")
                    except Exception:
                        continue
                
                if all_text:
                    df = pd.DataFrame([{
                        'raw_text': '\n\n'.join(all_text),
                        'word_count': len(' '.join(all_text).split()),
                        'pages_processed': len(all_text)
                    }])
                    
                    source_info = {
                        'source_file': file_path.name,
                        'source_type': 'pdf',
                        'extraction_method': 'ocr',
                        'pages_processed': len(pages),
                        'file_size': file_path.stat().st_size,
                        'file_hash': create_file_hash(file_path),
                        'thread_name': thread_name
                    }
                    
                    df = clean_dataframe(df, source_info)
                    frames.append(df)
                    
                    stats['methods'].append({'method': 'ocr', 'pages': len(all_text)})
                    
            except Exception as e:
                logger.debug(f"[{thread_name}] OCR failed: {e}")
        
        if frames:
            stats['status'] = 'success'
            stats['total_rows'] = sum(len(df) for df in frames)
            stats['processing_time'] = time.time() - start_time
            logger.info(f"[{thread_name}] ✅ PDF {file_path.name}: {len(frames)} tables, {stats['total_rows']} rows ({stats['processing_time']:.1f}s)")
        else:
            stats['error'] = 'No data extracted'
            
    except TimeoutError:
        stats['error'] = f'Processing timeout ({PDF_TIMEOUT}s)'
        logger.warning(f"[{thread_name}] ⏰ PDF {file_path.name}: Timeout")
    except Exception as e:
        stats['error'] = str(e)
        logger.error(f"[{thread_name}] ❌ PDF error {file_path.name}: {e}")
    
    return file_path.name, frames, stats

# ====== PARALLEL PROCESSING ORCHESTRATOR ======
def process_files_parallel(files: List[Path], progress_tracker: ProgressTracker) -> Tuple[List[pd.DataFrame], List[Dict]]:
    """Process files in parallel using thread and process pools"""
    
    # Group files by type for optimal processing
    csv_files = [f for f in files if f.suffix.lower() == '.csv']
    excel_files = [f for f in files if f.suffix.lower() in ['.xlsx', '.xls']]
    pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
    
    all_frames = []
    all_stats = []
    
    logger.info(f"File distribution: {len(csv_files)} CSV, {len(excel_files)} Excel, {len(pdf_files)} PDF")
    
    # Process CSV and Excel files with ThreadPoolExecutor (I/O bound)
    if csv_files or excel_files:
        logger.info(f"Processing {len(csv_files + excel_files)} CSV/Excel files with {MAX_WORKERS} threads...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit CSV tasks
            future_to_file = {}
            for file_path in csv_files:
                future = executor.submit(process_single_csv, file_path)
                future_to_file[future] = file_path
            
            # Submit Excel tasks
            for file_path in excel_files:
                future = executor.submit(process_single_excel, file_path)
                future_to_file[future] = file_path
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    file_name, frames, stats = future.result(timeout=60)  # 1 minute timeout per file
                    all_frames.extend(frames)
                    all_stats.append(stats)
                    progress_tracker.update(success=bool(frames))
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    all_stats.append({
                        'file': file_path.name,
                        'status': 'error',
                        'error': str(e)
                    })
                    progress_tracker.update(success=False)
    
    # Process PDF files with ProcessPoolExecutor (CPU bound)
    if pdf_files:
        logger.info(f"Processing {len(pdf_files)} PDF files with {min(MAX_WORKERS, 4)} processes...")
        
        # Limit concurrent PDF processes to prevent memory issues
        pdf_workers = min(MAX_WORKERS, 4)
        
        with ProcessPoolExecutor(max_workers=pdf_workers) as executor:
            future_to_file = {}
            for file_path in pdf_files:
                future = executor.submit(process_single_pdf, file_path)
                future_to_file[future] = file_path
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    file_name, frames, stats = future.result(timeout=PDF_TIMEOUT + 60)
                    all_frames.extend(frames)
                    all_stats.append(stats)
                    progress_tracker.update(success=bool(frames))
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Failed to process PDF {file_path.name}: {e}")
                    all_stats.append({
                        'file': file_path.name,
                        'type': 'pdf',
                        'status': 'error',
                        'error': str(e)
                    })
                    progress_tracker.update(success=False)
    
    return all_frames, all_stats

# ====== SCHEMA ANALYSIS ======
def analyze_schema(master_df: pd.DataFrame) -> Dict:
    """Analyze the schema of the master dataset"""
    schema = {
        'total_records': len(master_df),
        'total_columns': len(master_df.columns),
        'columns': {},
        'source_files': master_df['source_file'].value_counts().to_dict() if 'source_file' in master_df.columns else {},
        'data_types': {},
        'null_analysis': {},
        'unique_values_sample': {}
    }
    
    for col in master_df.columns:
        col_data = master_df[col]
        
        schema['columns'][col] = {
            'non_null_count': col_data.notna().sum(),
            'null_count': col_data.isna().sum(),
            'unique_count': col_data.nunique(),
            'data_type': str(col_data.dtype)
        }
        
        # Sample unique values (first 10)
        unique_vals = col_data.dropna().unique()[:10].tolist()
        schema['unique_values_sample'][col] = [str(v) for v in unique_vals]
        
        # Null percentage
        null_pct = (col_data.isna().sum() / len(col_data)) * 100
        schema['null_analysis'][col] = round(null_pct, 2)
    
    return schema

# ====== MAIN PARALLEL BUILDER ======
def build_master():
    """Enhanced parallel master data builder"""
    logger.info("=" * 80)
    logger.info("STARTING PARALLEL ENHANCED MASTER DATA BUILDER")
    logger.info("=" * 80)
    logger.info(f"System: {cpu_count()} CPU cores, Max workers: {MAX_WORKERS}")
    logger.info(f"Memory limit per process: {MEMORY_LIMIT_MB}MB")
    logger.info(f"PDF timeout: {PDF_TIMEOUT}s")
    
    # Backup existing files
    if MASTER_CSV.exists():
        backup_path = BACKUP_DIR / f"master_data_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        MASTER_CSV.rename(backup_path)
        logger.info(f"Backed up existing master data to {backup_path}")
    
    # Scan for files
    files = []
    for pattern in ["**/*.csv", "**/*.xlsx", "**/*.xls", "**/*.pdf"]:
        files.extend(DATA_DIR.glob(pattern))
    
    logger.info(f"Found {len(files)} files to process")
    
    if not files:
        logger.error("No supported files found in data directory")
        return
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(len(files))
    
    # Process files in parallel
    start_time = time.time()
    logger.info(f"\n{'='*60}")
    logger.info("STARTING PARALLEL PROCESSING")
    logger.info(f"{'='*60}")
    
    all_frames, all_stats = process_files_parallel(files, progress_tracker)
    
    processing_time = time.time() - start_time
    
    if not all_frames:
        logger.error("No data was successfully extracted from any file")
        return
    
    # Combine all frames
    logger.info(f"\n{'='*50}")
    logger.info("COMBINING DATA")
    logger.info(f"{'='*50}")
    
    logger.info(f"Combining {len(all_frames)} DataFrames...")
    
    # Use concat with better handling of mismatched columns
    master_df = pd.concat(all_frames, ignore_index=True, sort=False)
    
    # Remove duplicate rows but keep source information
    logger.info(f"Before deduplication: {len(master_df)} rows")
    
    # Create a hash column for deduplication
    metadata_cols = ['source_file', 'source_type', 'processed_at', 'row_id', 'file_hash', 
                     'sheet_name', 'page_number', 'table_number', 'thread_name']
    data_cols = [col for col in master_df.columns if col not in metadata_cols]
    
    if data_cols:
        master_df['data_hash'] = master_df[data_cols].astype(str).apply(
            lambda x: hashlib.md5(''.join(x.values).encode()).hexdigest(), axis=1
        )
        
        master_df = master_df.drop_duplicates(subset=['data_hash'], keep='first')
        logger.info(f"After deduplication: {len(master_df)} rows")
    
    # Generate schema analysis
    logger.info("Analyzing schema...")
    schema = analyze_schema(master_df)
    
    # Save outputs
    logger.info(f"\n{'='*40}")
    logger.info("SAVING OUTPUTS")
    logger.info(f"{'='*40}")
    
    try:
        master_df.to_csv(MASTER_CSV, index=False, encoding='utf-8')
        logger.info(f"✅ Saved CSV: {MASTER_CSV}")
    except Exception as e:
        logger.error(f"❌ Failed to save CSV: {e}")
    
    try:
        master_df.to_parquet(MASTER_PARQUET, index=False)
        logger.info(f"✅ Saved Parquet: {MASTER_PARQUET}")
    except Exception as e:
        logger.error(f"❌ Failed to save Parquet: {e}")
    
    # Calculate detailed statistics
    successful_stats = [s for s in all_stats if s.get('status') == 'success']
    failed_stats = [s for s in all_stats if s.get('status') != 'success']
    
    # Save metadata
    metadata = {
        'created_at': datetime.datetime.now().isoformat(),
        'processing_time_seconds': processing_time,
        'total_records': len(master_df),
        'total_columns': len(master_df.columns),
        'processing_stats': {
            'total_files': len(files),
            'successful': len(successful_stats),
            'failed': len(failed_stats),
            'success_rate': (len(successful_stats) / len(files)) * 100,
            'files_per_second': len(files) / processing_time,
            'records_per_second': len(master_df) / processing_time
        },
        'system_info': {
            'cpu_cores': cpu_count(),
            'max_workers': MAX_WORKERS,
            'pdf_timeout': PDF_TIMEOUT
        },
        'file_details': all_stats,
        'column_list': list(master_df.columns),
        'files_processed': [f.name for f in files]
    }
    
    try:
        with open(META_JSON, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved metadata: {META_JSON}")
    except Exception as e:
        logger.error(f"❌ Failed to save metadata: {e}")
    
    # Save schema analysis
    try:
        with open(SCHEMA_JSON, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved schema: {SCHEMA_JSON}")
    except Exception as e:
        logger.error(f"❌ Failed to save schema: {e}")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("PARALLEL PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"⏱️  Total Processing Time: {processing_time/60:.1f} minutes ({processing_time:.1f}s)")
    logger.info(f"📊 Total Records: {len(master_df):,}")
    logger.info(f"📋 Total Columns: {len(master_df.columns):,}")
    logger.info(f"✅ Successful Files: {len(successful_stats)}/{len(files)} ({(len(successful_stats)/len(files)*100):.1f}%)")
    logger.info(f"❌ Failed Files: {len(failed_stats)}/{len(files)} ({(len(failed_stats)/len(files)*100):.1f}%)")
    logger.info(f"🚀 Processing Speed: {len(files)/processing_time:.1f} files/second")
    logger.info(f"📈 Data Throughput: {len(master_df)/processing_time:.0f} records/second")
    
    # File type breakdown
    csv_success = len([s for s in successful_stats if s.get('type') == 'csv'])
    excel_success = len([s for s in successful_stats if s.get('type') == 'excel'])
    pdf_success = len([s for s in successful_stats if s.get('type') == 'pdf'])
    
    logger.info(f"📄 File Type Success:")
    logger.info(f"   - CSV: {csv_success}/{len([f for f in files if f.suffix.lower() == '.csv'])}")
    logger.info(f"   - Excel: {excel_success}/{len([f for f in files if f.suffix.lower() in ['.xlsx', '.xls']])}")
    logger.info(f"   - PDF: {pdf_success}/{len([f for f in files if f.suffix.lower() == '.pdf'])}")
    
    logger.info(f"💾 Output Files:")
    logger.info(f"   - CSV: {MASTER_CSV}")
    logger.info(f"   - Parquet: {MASTER_PARQUET}")
    logger.info(f"   - Metadata: {META_JSON}")
    logger.info(f"   - Schema: {SCHEMA_JSON}")
    logger.info(f"📁 Log File: {log_file}")
    
    # Memory usage estimation
    memory_mb = master_df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"💾 Final Dataset Memory Usage: {memory_mb:.1f} MB")
    
    print("\n" + "="*80)
    print("🎉 PARALLEL MASTER DATABASE CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"⚡ Processed {len(files)} files in {processing_time/60:.1f} minutes")
    print(f"📊 {len(master_df):,} records across {len(master_df.columns):,} columns")
    print(f"✅ {len(successful_stats)}/{len(files)} files processed successfully ({(len(successful_stats)/len(files)*100):.1f}%)")
    print(f"🚀 Average speed: {len(files)/processing_time:.1f} files/second")
    print(f"💾 Files saved in: {OUT_DIR}")
    print(f"📋 Check {META_JSON} for detailed statistics")
    print(f"📝 Full log: {log_file}")
    
    if failed_stats:
        print(f"\n⚠️  {len(failed_stats)} files failed to process:")
        for stat in failed_stats[:5]:  # Show first 5 failures
            print(f"   - {stat['file']}: {stat.get('error', 'Unknown error')}")
        if len(failed_stats) > 5:
            print(f"   ... and {len(failed_stats) - 5} more (see log for details)")

# ====== BATCH PROCESSING FOR VERY LARGE DATASETS ======
def build_master_batch(batch_size: int = 50):
    """Process files in batches for very large datasets (1000+ files)"""
    logger.info("=" * 80)
    logger.info("STARTING BATCH PROCESSING MODE")
    logger.info("=" * 80)
    
    # Scan all files
    files = []
    for pattern in ["**/*.csv", "**/*.xlsx", "**/*.xls", "**/*.pdf"]:
        files.extend(DATA_DIR.glob(pattern))
    
    if not files:
        logger.error("No supported files found")
        return
    
    logger.info(f"Found {len(files)} files - Processing in batches of {batch_size}")
    
    # Split into batches
    file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    all_master_frames = []
    overall_stats = []
    
    for batch_num, batch_files in enumerate(file_batches, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING BATCH {batch_num}/{len(file_batches)} ({len(batch_files)} files)")
        logger.info(f"{'='*60}")
        
        progress_tracker = ProgressTracker(len(batch_files))
        batch_frames, batch_stats = process_files_parallel(batch_files, progress_tracker)
        
        if batch_frames:
            # Combine batch frames
            batch_master = pd.concat(batch_frames, ignore_index=True, sort=False)
            all_master_frames.append(batch_master)
            logger.info(f"Batch {batch_num} completed: {len(batch_master)} records")
        
        overall_stats.extend(batch_stats)
        
        # Optional: Save intermediate results
        if batch_num % 5 == 0:  # Every 5 batches
            intermediate_path = OUT_DIR / f"intermediate_batch_{batch_num}.parquet"
            if all_master_frames:
                intermediate_df = pd.concat(all_master_frames, ignore_index=True, sort=False)
                intermediate_df.to_parquet(intermediate_path, index=False)
                logger.info(f"Saved intermediate results: {intermediate_path}")
    
    # Final combination
    if all_master_frames:
        logger.info(f"\nCombining {len(all_master_frames)} batch results...")
        final_master = pd.concat(all_master_frames, ignore_index=True, sort=False)
        
        # Deduplication
        metadata_cols = ['source_file', 'source_type', 'processed_at', 'row_id', 'file_hash', 
                         'sheet_name', 'page_number', 'table_number', 'thread_name']
        data_cols = [col for col in final_master.columns if col not in metadata_cols]
        
        if data_cols:
            final_master['data_hash'] = final_master[data_cols].astype(str).apply(
                lambda x: hashlib.md5(''.join(x.values).encode()).hexdigest(), axis=1
            )
            final_master = final_master.drop_duplicates(subset=['data_hash'], keep='first')
        
        # Save final results
        final_master.to_csv(MASTER_CSV, index=False, encoding='utf-8')
        final_master.to_parquet(MASTER_PARQUET, index=False)
        
        # Save metadata
        successful_stats = [s for s in overall_stats if s.get('status') == 'success']
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'processing_mode': 'batch',
            'batch_size': batch_size,
            'total_batches': len(file_batches),
            'total_records': len(final_master),
            'total_columns': len(final_master.columns),
            'processing_stats': {
                'total_files': len(files),
                'successful': len(successful_stats),
                'failed': len(files) - len(successful_stats)
            },
            'file_details': overall_stats
        }
        
        with open(META_JSON, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print(f"📊 {len(final_master):,} records from {len(files)} files")
        print(f"✅ {len(successful_stats)} successful, {len(files) - len(successful_stats)} failed")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        build_master_batch(batch_size)
    else:
        build_master()