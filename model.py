import pandas as pd
import re

def parse_soft_annotation(soft_file_path):
    """Parse GEO SOFT file format to extract probe-gene mappings"""
    
    print("Parsing SOFT file...")
    
    with open(soft_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into lines
    lines = content.split('\n')
    
    # Find where the data table begins
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith('!platform_table_begin'):
            table_start = i + 1
            break
    
    if table_start is None:
        print("Error: Could not find the start of the data table")
        return None
    
    # Find where the data table ends
    table_end = None
    for i in range(table_start, len(lines)):
        if lines[i].startswith('!platform_table_end'):
            table_end = i
            break
    
    if table_end is None:
        # If no end marker, read until end of file
        table_end = len(lines)
    
    # Extract the header (column names)
    header_line = lines[table_start]
    columns = header_line.split('\t')
    print(f"Found columns: {columns}")
    
    # Extract the data rows
    data_rows = []
    for i in range(table_start + 1, table_end):
        line = lines[i].strip()
        if line and not line.startswith('!'):
            data_rows.append(line.split('\t'))
    
    # Create DataFrame
    annotation_df = pd.DataFrame(data_rows, columns=columns)
    
    print(f"Successfully parsed {len(annotation_df)} probes")
    
    return annotation_df

def clean_annotation_data(annotation_df):
    """Clean and prepare the annotation data for gene mapping"""
    
    # Display available columns
    print("\nAvailable columns in annotation:")
    for col in annotation_df.columns:
        print(f"  - {col}")
    
    # Look for gene-related columns
    # Common patterns for gene symbol columns
    gene_col_candidates = []
    for col in annotation_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['gene', 'symbol', 'name', 'definition']):
            gene_col_candidates.append(col)
    
    print(f"\nPotential gene columns: {gene_col_candidates}")
    
    # Let's examine the data to find the best gene column
    if gene_col_candidates:
        print("\nSample data from potential gene columns:")
        for col in gene_col_candidates[:3]:  # Check first 3 candidates
            unique_vals = annotation_df[col].dropna().unique()
            print(f"\n{col} (first 5 unique values):")
            for val in unique_vals[:5]:
                print(f"  - {val}")
    
    return annotation_df

# Parse the SOFT file
soft_file_path = 'GPL4381_family.soft'  # Update this with your actual file name/path
annotation_df = parse_soft_annotation(soft_file_path)

if annotation_df is not None:
    # Clean the data
    annotation_df = clean_annotation_data(annotation_df)
    
    # Save as CSV for future use
    annotation_df.to_csv('GPL4381_annotation_parsed.csv', index=False)
    print(f"\nParsed annotation saved as 'GPL4381_annotation_parsed.csv'")
    
    # Let's try to identify the best column for gene symbols
    # Based on common GEO patterns, GB_DEFINITION often contains gene names
    if 'GB_DEFINITION' in annotation_df.columns:
        print("\nExtracting gene symbols from GB_DEFINITION...")
        
        def extract_gene_symbol(definition):
            if pd.isna(definition):
                return None
            # Common pattern: "Homo sapiens [gene name], mRNA"
            match = re.search(r'Homo sapiens ([\w\-]+)', str(definition))
            if match:
                return match.group(1)
            return None
        
        annotation_df['Gene_Symbol'] = annotation_df['GB_DEFINITION'].apply(extract_gene_symbol)
        
        # Count how many genes we successfully extracted
        genes_extracted = annotation_df['Gene_Symbol'].notna().sum()
        print(f"Successfully extracted gene symbols for {genes_extracted} probes")
        
        # Show some examples
        print("\nSample gene symbol mappings:")
        sample_data = annotation_df[annotation_df['Gene_Symbol'].notna()].head(10)
        for _, row in sample_data.iterrows():
            print(f"  Probe {row['ID']} -> {row['Gene_Symbol']}")
    
    # Also check GB_ACC for accession numbers that might help
    if 'GB_ACC' in annotation_df.columns:
        print(f"\nGB_ACC column sample: {annotation_df['GB_ACC'].dropna().unique()[:5]}")
    
    # Save the enhanced annotation
    annotation_df.to_csv('GPL4381_annotation_enhanced.csv', index=False)
    print("\nEnhanced annotation saved!")

else:
    print("Failed to parse the SOFT file")

# If automatic parsing doesn't work well, let's try a manual approach
def manual_gene_mapping_strategy(expression_data):
    """Alternative strategy if automatic gene mapping fails"""
    print("\n" + "="*50)
    print("ALTERNATIVE STRATEGY: Direct matching")
    print("="*50)
    
    # The 18 key genes we're looking for
    key_genes = [
        'C1QTNF3', 'CA11', 'CD9', 'CDK5RAP3', 'CLDN1', 'DMC1', 'EYA1', 'IFI44L',
        'KNG1', 'KREMEN1', 'NT5DC2', 'NTRK3', 'PSG4', 'RFX4', 'RHOBTB3', 
        'SLC1A2', 'SMARCA2', 'TRIM58'
    ]
    
    print("Since we have the probe sequences, we could:")
    print("1. Use BLAST to match probe sequences to genes")
    print("2. Search for these specific 18 genes in public databases")
    print("3. Use the paper's supplementary data if available")
    
    return key_genes

# Check if we have the annotation working
if annotation_df is not None and 'Gene_Symbol' in annotation_df.columns:
    print("\n✅ Gene annotation ready! We can proceed to map probes to genes.")
else:
    print("\n⚠️  Gene annotation incomplete. We may need alternative approaches.")
    manual_gene_mapping_strategy(None)