#!/usr/bin/env python3
"""
Export Pydantic schemas to JSON Schema files for CLI usage.

This script generates JSON Schema files from Pydantic models,
making it easy to use schemas via the schema_file argument.
"""

#!/usr/bin/env python3
"""
Export Pydantic schemas to JSON Schema files for CLI usage.

This script generates JSON Schema files from Pydantic models,
making it easy to use schemas via the schema_file argument.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.medical_qa_schemas import (
    PUBMEDQA_SCHEMAS,
    MULTIPLE_CHOICE_SCHEMAS,
    SCHEMA_METADATA,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "json_schemas"
OUTPUT_DIR.mkdir(exist_ok=True)


def export_schema(schema_class, output_path: Path, metadata: dict = None):
    """Export a Pydantic schema to JSON Schema format."""
    schema_dict = schema_class.model_json_schema()
    
    # Add metadata if provided
    if metadata:
        schema_dict["_metadata"] = metadata
    
    # Write to file
    with open(output_path, "w") as f:
        json.dump(schema_dict, f, indent=2)
    
    print(f"✅ Exported: {output_path.name}")


def main():
    """Export all schemas to JSON files."""
    print("Exporting Pydantic schemas to JSON Schema files...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Export PubMedQA schemas
    print("📋 PubMedQA Schemas:")
    for name, schema_class in PUBMEDQA_SCHEMAS.items():
        metadata = SCHEMA_METADATA.get(name, {})
        output_path = OUTPUT_DIR / f"pubmedqa_{name}.json"
        export_schema(schema_class, output_path, metadata)
    
    print()
    
    # Export Multiple Choice schemas
    print("📋 Multiple Choice Schemas:")
    for name, schema_class in MULTIPLE_CHOICE_SCHEMAS.items():
        output_path = OUTPUT_DIR / f"mc_{name}.json"
        export_schema(schema_class, output_path)
    
    print()
    print(f"✨ All schemas exported to: {OUTPUT_DIR}")
    print("\nUsage example:")
    print(f"  --model_args schema_file={OUTPUT_DIR}/pubmedqa_level1_strict.json")


if __name__ == "__main__":
    main()

