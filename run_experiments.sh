#!/bin/bash

# Base configuration file to copy for each experiment
BASE_CONFIG_FILE="configuration/unet_config.yaml"

# Output directory for experiments (only used for naming in Python)
BASE_OUTPUT_NAME="unet"

# Define the age skew and gender skew scenarios with specified order for age skew
declare -A AGE_SKEWS=(
    ["100_young_0_old"]="1.0,0.0"
    ["100_young_25_old"]="1.0,0.25"
    ["100_young_50_old"]="1.0,0.5"
    ["100_young_75_old"]="1.0,0.75"
    ["100_young_100_old"]="1.0,1.0"
    ["100_old_25_young"]="0.25,1.0"
    ["100_old_50_young"]="0.5,1.0"
    ["100_old_75_young"]="0.75,1.0"
    ["100_old_0_young"]="0.0,1.0"
)

declare -A GENDER_SKEWS=(
    ["100_male_0_female"]="1.0,0.0"
    ["100_male_25_female"]="1.0,0.25"
    ["100_male_50_female"]="1.0,0.5"
    ["100_male_75_female"]="1.0,0.75"
    ["100_male_100_female"]="1.0,1.0"
    ["100_female_75_male"]="0.75,1.0"
    ["100_female_50_male"]="0.5,1.0"
    ["100_female_25_male"]="0.25,1.0"
    ["100_female_0_male"]="0.0,1.0"
)

declare -A TYPE_SKEWS=(
    ["100_glioblastoma_0_other"]="1.0,0.0"
    ["100_glioblastoma_25_other"]="1.0,0.25"
    ["100_glioblastoma_50_other"]="1.0,0.5"
    ["100_glioblastoma_75_other"]="1.0,0.75"
    ["100_glioblastoma_100_other"]="1.0,1.0"
    ["100_other_75_glioblastoma"]="0.75,1.0"
    ["100_other_50_glioblastoma"]="0.5,1.0"
    ["100_other_25_glioblastoma"]="0.25,1.0"
    ["100_other_0_glioblastoma"]="0.0,1.0"
)

# Function to run each experiment with a unique configuration file
run_experiment() {
    local skew_type=$1
    local skew_name=$2
    local skew_values=$3

    # Create a copy of the base config file for this experiment
    EXPERIMENT_CONFIG="configuration/unet_config_${skew_type}_${skew_name}.yaml"
    cp "$BASE_CONFIG_FILE" "$EXPERIMENT_CONFIG"

    # Set unique output name for each experiment
    local output_name="${BASE_OUTPUT_NAME}-${skew_name}"

    # Use Python to modify the YAML configuration file
    python3 - <<END
import yaml

# Load the YAML configuration
with open("$EXPERIMENT_CONFIG", 'r') as file:
    config = yaml.safe_load(file)

# Update configuration based on skew type
config['output_name'] = "$output_name"
if "$skew_type" == "age":
    config['age_skew'] = [$skew_values]
    config['gender_skew'] = None  # Clear gender skew for age experiments
    config['ttype_skew'] = None  # Clear type skew for age experiments
elif "$skew_type" == "gender":
    config['gender_skew'] = [$skew_values]
    config['age_skew'] = None  # Clear age skew for gender experiments
    config['ttype_skew'] = None  # Clear
elif "$skew_type" == "type":
    config['ttype_skew'] = [$skew_values]
    config['age_skew'] = None  # Clear age skew for
    config['gender_skew'] = None

# Save the updated configuration back to the file
with open("$EXPERIMENT_CONFIG", 'w') as file:
    yaml.safe_dump(config, file)
END

    # Confirm output name in the file
    echo "Running experiment: ${skew_type} skew with configuration: ${skew_name}"
    echo "Configured output_name: $output_name"

    # Execute the algorithm with this specific configuration file
    python train_reconstruction.py -c "$EXPERIMENT_CONFIG"
    
    # Optional: Clean up after each run if you don’t need the individual config files
    # rm "$EXPERIMENT_CONFIG"
}

# Run age skew experiments
for skew_name in "${!AGE_SKEWS[@]}"; do
    run_experiment "age" "$skew_name" "${AGE_SKEWS[$skew_name]}"
done

# Run gender skew experiments
for skew_name in "${!GENDER_SKEWS[@]}"; do
    run_experiment "gender" "$skew_name" "${GENDER_SKEWS[$skew_name]}"
done

# Run type skew experiments
for skew_name in "${!TYPE_SKEWS[@]}"; do
    run_experiment "type" "$skew_name" "${TYPE_SKEWS[$skew_name]}"
done
