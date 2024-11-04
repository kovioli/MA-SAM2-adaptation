import os
import csv
import pandas as pd
from pathlib import Path


def get_particle_name(particle_id, mapping):
    """Get particle name from particle ID using the mapping."""
    for item in mapping:
        if item["particle_id"] == particle_id:
            return item["particle_name"]
    return None


def process_motl_files(particle_id_mapping):
    """Process all MOTL files for given particle IDs and generate coordinate output."""
    # Create output directory if it doesn't exist
    output_dir = Path("PARTICLE_COORDS")
    output_dir.mkdir(exist_ok=True)

    # Open output file
    with open(output_dir / "deepict_grandmodel.txt", "w") as outfile:
        # Process each particle ID
        for particle_info in particle_id_mapping:
            particle_id = particle_info["particle_id"]
            particle_name = particle_info["particle_name"]

            # Construct the pattern for glob
            pattern = f"motl_*.csv"

            # Search for matching files
            base_path = f"/media/hdd1/oliver/DeePiCt/PREDICT/predictions/shrec_p{particle_id:02d}_grandmodel_best/model_9_p{particle_id}_grandmodel/ribo"
            base_path = Path(base_path)
            matching_files = list(base_path.glob(pattern))
            print("MATCHING FILES:", matching_files)

            # Process each file
            for file_path in matching_files:
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path, header=None)

                    # Extract coordinates (columns 7,8,9 - remember Python is 0-based)
                    coordinates = df.iloc[:, [7, 8, 9]].values

                    # Write to output file
                    for coord in coordinates:
                        line = f"{particle_name} {int(coord[0])} {int(coord[1])} {int(coord[2])}\n"
                        outfile.write(line)

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")


def main():
    # Base path where the CSV files are located
    base_path = "/media/hdd1/oliver/DeePiCt/PREDICT/predictions/shrec_p01_grandmodel_best/model_9_p1_grandmodel/ribo"

    # Particle ID mapping
    particle_id_mapping = [
        {"particle_id": 1, "particle_name": "3cf3"},
        {"particle_id": 2, "particle_name": "1s3x"},
        {"particle_id": 3, "particle_name": "1u6g"},
        {"particle_id": 4, "particle_name": "4cr2"},
        {"particle_id": 5, "particle_name": "1qvr"},
        {"particle_id": 6, "particle_name": "3h84"},
        {"particle_id": 7, "particle_name": "2cg9"},
        {"particle_id": 8, "particle_name": "3qm1"},
        {"particle_id": 9, "particle_name": "3gl1"},
        {"particle_id": 10, "particle_name": "3d2f"},
        {"particle_id": 11, "particle_name": "4d8q"},
        {"particle_id": 12, "particle_name": "1bxn"},
    ]

    process_motl_files(particle_id_mapping)
    print("Processing complete. Output saved to PARTICLE_COORDS/deepict_grandmodel.txt")


if __name__ == "__main__":
    main()
