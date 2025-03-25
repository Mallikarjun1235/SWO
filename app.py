from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import io
import json
import os
from datetime import datetime



app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
CSV_FILE_PATH = "website_configurations.csv"

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Constants
ALLOWED_APPLICATIONS = ["EBHS", "EBS", "EmBHS", "DAWN"]
ALLOWED_MATCHING_MODELS = ["epm1dot1tt", "epm1dot1", "epm1dot1small", 
                         "epmbasevariant", "epmbaseexact", "vibe"]

# Required columns for validation
SWO_REQUIRED_COLUMNS = [
    "sourceId", "sourceName", "marketplaceIds", "programName", "sdoProductStrategy"
]

METRICS_REQUIRED_COLUMNS = [
    "sourceId", "marketplaceIds", "application", "matchingModel",
    "SDQ_score", "Determinism", "IF_Accuracy", "INF_Accuracy"
]

def validate_numeric_percentage(value, column_name):
    """Convert and validate percentage values"""
    try:
        if isinstance(value, str):
            value = float(value.strip('%'))
        else:
            value = float(value)

        if value < 1:
            value *= 100

        if not (0 <= value <= 100):
            raise ValueError(f"{column_name} must be between 0 and 100")
        
        return value
    except ValueError as e:
        raise ValueError(f"Invalid {column_name}: {str(e)}")

def validate_smp_match(swo_df, metrics_df):
    """Validate that all sourceId-marketplaceIds combinations exist in both files"""
    # Create SMP columns for both dataframes
    swo_df['SMP'] = swo_df['sourceId'].astype(str) + "_" + swo_df['marketplaceIds'].astype(str)
    metrics_df['SMP'] = metrics_df['sourceId'].astype(str) + "_" + metrics_df['marketplaceIds'].astype(str)
    
    # Get sets of SMPs from both files
    swo_smps = set(swo_df['SMP'])
    metrics_smps = set(metrics_df['SMP'])
    
    error_messages = []
    
    # Check for SMPs missing in metrics file
    missing_in_metrics = swo_smps - metrics_smps
    if missing_in_metrics:
        error_messages.append("Following sourceId and marketplaceIds combinations missing in Metrics file:")
        for smp in missing_in_metrics:
            source_id, mp = smp.split('_')
            error_messages.append(f"sourceId: {source_id}, MP: {mp}")
    
    # Check for SMPs missing in SWO file
    missing_in_swo = metrics_smps - swo_smps
    if missing_in_swo:
        if error_messages:  # Add blank line if there are previous errors
            error_messages.append("")
        error_messages.append("Following sourceId and marketplaceIds combinations missing in SWO file:")
        for smp in missing_in_swo:
            source_id, mp = smp.split('_')
            error_messages.append(f"sourceId: {source_id}, MP: {mp}")
    
    if error_messages:
        raise ValueError("\n".join(error_messages))

def validate_swo_file(df):
    """Validate SWO input file"""
    # First check sourceId and marketplaceIds for NA/empty values
    key_errors = []
    
    # Check sourceId
    empty_source_ids = df[df['sourceId'].isnull() | (df['sourceId'].astype(str).str.strip() == '')]
    if not empty_source_ids.empty:
        key_errors.append("sourceId cannot be empty")

    empty_source_name = df[df['sourceName'].isnull() | (df['sourceName'].astype(str).str.strip() == '')]
    if not empty_source_name.empty:
        key_errors.append("sourceName cannot be empty")    
        
    # Check marketplaceIds
    empty_mps = df[df['marketplaceIds'].isnull() | (df['marketplaceIds'].astype(str).str.strip() == '')]
    if not empty_mps.empty:
        key_errors.append("marketplaceIds cannot be empty")
        
    if key_errors:
        raise ValueError("\n".join(key_errors))
    
    # Then check for duplicate SMPs
    df['SMP'] = df['sourceId'].astype(str) + "_" + df['marketplaceIds'].astype(str)
    duplicates = df[df['SMP'].duplicated()]
    if not duplicates.empty:
        dup_errors = ["Duplicate entries found for following sourceId and marketplaceIds combinations:"]
        for _, row in duplicates.iterrows():
            dup_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}")
        raise ValueError("\n".join(dup_errors))

    # Continue with other validations...
    program_errors = []
    strategy_errors = []
    
    # Validate programName values with space check
    allowed_programs = ["SourceInsights", "SelectionGapClosure"]
    for _, row in df.iterrows():
        # Check for empty values first
        if pd.isna(row['programName']) or str(row['programName']).strip() == '':
            program_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, programName cannot be empty")
        else:
            program = str(row['programName'])
            if program.strip() != program:
                program_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, programName '{program}' contains leading or trailing spaces")
            elif program.strip() not in allowed_programs:
                program_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, Invalid programName: {program}")
    
    # Validate sdoProductStrategy values with space check
    allowed_strategies = ["detail", "browse"]
    for _, row in df.iterrows():
        # Check for empty values first
        if pd.isna(row['sdoProductStrategy']) or str(row['sdoProductStrategy']).strip() == '':
            strategy_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, sdoProductStrategy cannot be empty")
        else:
            strategy = str(row['sdoProductStrategy'])
            if strategy.strip() != strategy:
                strategy_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, sdoProductStrategy '{strategy}' contains leading or trailing spaces")
            elif strategy.strip() not in allowed_strategies:
                strategy_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, Invalid sdoProductStrategy: {strategy}")
    
    error_messages = []
    
    if program_errors:
        error_messages.append("Program Name Errors:")
        error_messages.extend(program_errors)
        error_messages.append(f"Allowed programNames are: {', '.join(allowed_programs)}")
    
    if strategy_errors:
        if program_errors:  # Add a blank line between error types
            error_messages.append("")
        error_messages.append("SDO Product Strategy Errors:")
        error_messages.extend(strategy_errors)
        error_messages.append(f"Allowed sdoProductStrategy values are: {', '.join(allowed_strategies)}")
    
    if error_messages:
        raise ValueError("\n".join(error_messages))

def validate_metrics_file(df, is_brand):
    """Validate Metrics input file with brand/competitor specific validation"""
    # First check sourceId and marketplaceIds for NA/empty values
    key_errors = []
    
    # Check sourceId
    empty_source_ids = df[df['sourceId'].isnull() | (df['sourceId'].astype(str).str.strip() == '')]
    if not empty_source_ids.empty:
        key_errors.append("sourceId cannot be empty")   
        
    # Check marketplaceIds
    empty_mps = df[df['marketplaceIds'].isnull() | (df['marketplaceIds'].astype(str).str.strip() == '')]
    if not empty_mps.empty:
        key_errors.append("marketplaceIds cannot be empty")
        
    if key_errors:
        raise ValueError("\n".join(key_errors))
    
    # Then check for duplicate SMPs
    df['SMP'] = df['sourceId'].astype(str) + "_" + df['marketplaceIds'].astype(str)
    duplicates = df[df['SMP'].duplicated()]
    if not duplicates.empty:
        dup_errors = ["Duplicate entries found for following sourceId and marketplaceIds combinations:"]
        for _, row in duplicates.iterrows():
            dup_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}")
        raise ValueError("\n".join(dup_errors))

    # Check for required columns
    missing_columns = [col for col in METRICS_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in Metrics file: {', '.join(missing_columns)}")
    
    # Define required fields based on brand/competitor selection
    required_fields = ["sourceId", "marketplaceIds", "application", "matchingModel", 
                      "Determinism", "IF_Accuracy", "INF_Accuracy"]
    if not is_brand:
        required_fields.append("SDQ_score")
    
    error_messages = []
    
    # Check for blanks in required fields
    for col in required_fields:
        empty_rows = df[df[col].isnull() | (df[col].astype(str).str.strip() == '')]
        if not empty_rows.empty:
            empty_source_ids = empty_rows['sourceId'].tolist()
            error_messages.append(f"Column '{col}' contains empty values for sourceId(s): {empty_source_ids}")

    # Application validation
    application_errors = []
    invalid_applications = df[~df['application'].isin(ALLOWED_APPLICATIONS)]
    if not invalid_applications.empty:
        for _, row in invalid_applications.iterrows():
            application = str(row['application'])
            if application.strip() != application:
                application_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, application '{application}' contains leading or trailing spaces")
            else:
                application_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, invalid application: {application}")
    
    if application_errors:
        error_messages.append("\nApplication Errors:")
        error_messages.extend(application_errors)
        error_messages.append(f"Allowed applications are: {', '.join(ALLOWED_APPLICATIONS)}")

    # Matching Model validation
    model_errors = []
    invalid_models = df[~df['matchingModel'].isin(ALLOWED_MATCHING_MODELS)]
    if not invalid_models.empty:
        for _, row in invalid_models.iterrows():
            model = str(row['matchingModel'])
            if model.strip() != model:
                model_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, matchingModel '{model}' contains leading or trailing spaces")
            else:
                model_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, invalid matching model: {model}")
    
    if model_errors:
        if application_errors:  # Add a blank line between error types
            error_messages.append("")
        error_messages.append("Matching Model Errors:")
        error_messages.extend(model_errors)
        error_messages.append(f"Allowed matching models are: {', '.join(ALLOWED_MATCHING_MODELS)}")

    # Numeric field validation
    numeric_fields = ['Determinism', 'IF_Accuracy', 'INF_Accuracy']
    if not is_brand:
        numeric_fields.append('SDQ_score')
    
    numeric_errors = []
    for field in numeric_fields:
        for _, row in df.iterrows():
            try:
                value = str(row[field])
                if value.strip() != value:
                    numeric_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, {field} '{value}' contains leading or trailing spaces")
                    continue
                
                # Remove % if present and convert to float
                value = value.strip('%')
                value = float(value)
                
                # Check range
                if value < 0 or value > 100:
                    numeric_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, {field} must be between 0 and 100, found: {value}")
            except ValueError:
                numeric_errors.append(f"sourceId: {row['sourceId']}, MP: {row['marketplaceIds']}, Invalid {field} value: {row[field]}")
    
    if numeric_errors:
        if error_messages:  # Add a blank line if there are previous errors
            error_messages.append("")
        error_messages.append("Numeric Field Errors:")
        error_messages.extend(numeric_errors)
    
    if error_messages:
        raise ValueError("Validation errors in Metrics file:\n" + "\n".join(error_messages))
def create_smp_column(df):
    """Create SMP column by concatenating sourceId and marketplaceIds"""
    return df['sourceId'].astype(str) + "_" + df['marketplaceIds'].astype(str)

def generate_sdo_and_sci_product_strategy(program_name, application, is_source_qualified, 
                                        matching_model, deduped_pipeline, sdo_value):
    """Generate strategy JSON"""
    deduped_required = deduped_pipeline == 'true'
    
    sdo_product_strategy = {
        "grainOfExtraction": sdo_value
    }

    if program_name == "SourceInsights":
        sci_product_strategy = {
            "targetAmazonMarketplaceIds": [],
            "applications": [],
            "clustering": {
                "type": "SingleSource"
            }
        }
    else:
        sci_product_strategy = {
            "targetAmazonMarketplaceIds": [],
            "applications": [application] if application else [],
            "isSourceQualified": bool(is_source_qualified),
            "matching": {
                "model": matching_model
            },
            "clustering": {
                "dedupeRequired": deduped_required
            }
        }

    return (
        json.dumps(sdo_product_strategy, separators=(',', ':'), ensure_ascii=False),
        json.dumps(sci_product_strategy, separators=(',', ':'), ensure_ascii=False)
    )

def process_excel_files(swo_file, metrics_file, brand_or_competitor):
    """Process both SWO and Metrics files"""
    try:
        # Read Excel files
        swo_df = pd.read_excel(io.BytesIO(swo_file['content']))
        metrics_df = pd.read_excel(io.BytesIO(metrics_file['content']))

        # Clean column names
        swo_df.columns = swo_df.columns.str.strip()
        metrics_df.columns = metrics_df.columns.str.strip()

        # Validate files
        validate_swo_file(swo_df)
        validate_metrics_file(metrics_df, brand_or_competitor == 'Brand')

        # Validate SMP match between files
        validate_smp_match(swo_df, metrics_df)

        # Create SMP columns
        swo_df['SMP'] = create_smp_column(swo_df)
        metrics_df['SMP'] = create_smp_column(metrics_df)

        # Process data
        processed_data = []
        output_messages = []
        acbp_data = []
        low_dq_data = []

        # Merge dataframes
        merged_df = pd.merge(swo_df, metrics_df, on='SMP', suffixes=('_swo', '_metrics'))

        for _, row in merged_df.iterrows():
            try:
                determinism = validate_numeric_percentage(row['Determinism'], 'Determinism')
                accuracy = validate_numeric_percentage(row['IF_Accuracy'], 'IF_Accuracy')
                inf_accuracy = validate_numeric_percentage(row['INF_Accuracy'], 'INF_Accuracy')

                meets_deduper_conditions = (
                    determinism > 79 and 
                    accuracy > 89 and 
                    inf_accuracy > 89
                )

                if brand_or_competitor == 'Brand':
                    is_source_qualified = True
                    deduper = meets_deduper_conditions
                else:
                    sdq_score = validate_numeric_percentage(row['SDQ_score'], 'SDQ_score')
                    is_source_qualified = sdq_score >= 79
                    deduper = is_source_qualified and meets_deduper_conditions

                deduper_str = str(deduper).lower()

                # Generate strategies
                sdo_strategy, sci_strategy = generate_sdo_and_sci_product_strategy(
                    row['programName'],
                    row['application'],
                    is_source_qualified,
                    row['matchingModel'],
                    deduper_str,
                    row['sdoProductStrategy']
                )

                processed_data.append([
                    row['sourceId_swo'],
                    row['sourceName'],
                    row['programName'],
                    row['marketplaceIds_swo'],
                    sdo_strategy,
                    sci_strategy
                ])

                # Categorize data
                if deduper:
                    acbp_data.append({
                        'sourceId': row['sourceId_swo'],
                        'marketplaceIds': row['marketplaceIds_swo']
                    })
                else:
                    low_dq_data.append({
                        'sourceId': row['sourceId_swo'],
                        'marketplaceIds': row['marketplaceIds_swo']
                    })

                # Add processing details
                output_messages = [f"Successfully processed {len(processed_data)} configurations"]

            except Exception as e:
                output_messages.append(f"Error processing {row['sourceName']}: {str(e)}")
                continue

        # Save categorized data
        if acbp_data:
            pd.DataFrame(acbp_data).to_csv('ACBP.csv', index=False)
        if low_dq_data:
            pd.DataFrame(low_dq_data).to_csv('Low_DQ.csv', index=False)

        return processed_data, output_messages

    except Exception as e:
        raise ValueError(f"Error processing files: {str(e)}")

def save_to_csv(data):
    """Save the data to CSV file"""
    df = pd.DataFrame(data, columns=[
        "sourceId", "sourceName", "programName", "marketplaceIds", 
        "sdoProductStrategy", "sciProductStrategy"
    ])
    df.to_csv(CSV_FILE_PATH, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_user_input', methods=['POST'])
def submit_user_input():
    try:
        source_id = request.form['sourceId']
        source_name = request.form['sourceName']
        marketplace_ids = request.form['marketplaceIds']
        sdo_value = request.form['sdoProductStrategy']
        program_name = request.form['programName']
        
        application = request.form.get('application', '')
        source_dq_cleared = request.form.get('sourceDqCleared', '')
        matching_model = request.form.get('matchingModel', '')
        deduped_pipeline = request.form.get('dedupedPipeline', '')

        sdo_strategy, sci_strategy = generate_sdo_and_sci_product_strategy(
            program_name, application, source_dq_cleared, 
            matching_model, deduped_pipeline, sdo_value
        )

        data = [[
            source_id, source_name, program_name, marketplace_ids,
            sdo_strategy, sci_strategy
        ]]
        save_to_csv(data)

        return jsonify({
            'status': 'success',
            'messages': [
                f"Configuration generated for {source_name}",
                f"SDO Strategy: {sdo_strategy}",
                f"SCI Strategy: {sci_strategy}"
            ],
            'downloads': [{
                'url': '/download/configurations',
                'filename': 'website_configurations.csv'
            }]
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/submit_excel', methods=['POST'])
def submit_excel():
    try:
        if 'swoUpload' not in request.files or 'metricsUpload' not in request.files:
            return jsonify({'status': 'error', 'message': 'Both files are required'})

        brand_or_competitor = request.form.get('brandOrCompetitor', '')
        if not brand_or_competitor:
            return jsonify({'status': 'error', 'message': 'Please select Brand or Competitor'})

        swo_file = request.files['swoUpload']
        metrics_file = request.files['metricsUpload']

        swo_data = {
            'filename': secure_filename(swo_file.filename),
            'content': swo_file.read()
        }
        metrics_data = {
            'filename': secure_filename(metrics_file.filename),
            'content': metrics_file.read()
        }

        processed_data, output_messages = process_excel_files(
            swo_data, metrics_data, brand_or_competitor
        )

        if processed_data:
            save_to_csv(processed_data)

        downloads = [{
            'url': '/download/configurations',
            'filename': 'website_configurations.csv'
        }]

        if os.path.exists('ACBP.csv'):
            downloads.append({
                'url': '/download/acbp',
                'filename': 'ACBP.csv'
            })
        if os.path.exists('Low_DQ.csv'):
            downloads.append({
                'url': '/download/low_dq',
                'filename': 'Low_DQ.csv'
            })

        return jsonify({
            'status': 'success',
            'messages': output_messages,
            'downloads': downloads
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download/<file_type>')
def download_file(file_type):
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if file_type == 'configurations':
            return send_file(
                CSV_FILE_PATH,
                as_attachment=True,
                download_name=f'website_configurations_{timestamp}.csv')
        elif file_type == 'acbp':
            return send_file(
                'ACBP.csv',
                as_attachment=True,
                download_name=f'ACBP_{timestamp}.csv'
            )
        elif file_type == 'low_dq':
            return send_file(
                'Low_DQ.csv',
                as_attachment=True,
                download_name=f'Low_DQ_{timestamp}.csv'
            )
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type requested'
            }), 400

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error downloading file: {str(e)}'
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum file size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error occurred'
    }), 500

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_timestamp():
    """Get current timestamp for file naming"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def cleanup_old_files():
    """Clean up temporary files older than 24 hours"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (current_time - file_modified).days >= 1:
                os.remove(filepath)
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

# Additional utility functions for logging and error tracking
def log_error(error_message, error_type="ERROR"):
    """Log errors to a file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {error_type}: {error_message}\n"
    
    with open('error_log.txt', 'a') as log_file:
        log_file.write(log_message)

def validate_input_data(data):
    """Validate user input data"""
    required_fields = ['sourceId', 'sourceName', 'marketplaceIds', 'sdoProductStrategy', 'programName']
    
    for field in required_fields:
        if field not in data or not data[field]:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate marketplaceIds format (comma-separated numbers)
    marketplace_ids = data['marketplaceIds'].split(',')
    try:
        [int(mid.strip()) for mid in marketplace_ids]
    except ValueError:
        raise ValueError("Invalid marketplaceIds format. Must be comma-separated numbers.")

def initialize_app():
    """Initialize application settings and create required directories"""
    # Create required directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Create log file if it doesn't exist
    if not os.path.exists('error_log.txt'):
        open('error_log.txt', 'a').close()
    
    # Clean up old files on startup
    cleanup_old_files()

# Scheduled task for cleanup (runs every hour)

# Application initialization
if __name__ == '__main__':
    initialize_app()
    
    
    # Start the Flask application
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )