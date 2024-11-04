import os
import yaml
import subprocess

# Base configuration file to copy for each experiment
BASE_CONFIG_FILE = "configuration/unet_config.yaml"
BASE_OUTPUT_NAME = "unet"

# Define the age skew and gender skew scenarios
AGE_SKEWS = {
    "100_young_0_old": "1.0,0.0",
    "100_young_25_old": "1.0,0.25",
    "100_young_50_old": "1.0,0.5",
    "100_young_75_old": "1.0,0.75",
    "100_young_100_old": "1.0,1.0",
    "100_old_25_young": "0.25,1.0",
    "100_old_50_young": "0.5,1.0",
    "100_old_75_young": "0.75,1.0",
    "100_old_0_young": "0.0,1.0"
}

GENDER_SKEWS = {
    "100_male_0_female": "1.0,0.0",
    "100_male_25_female": "1.0,0.25",
    "100_male_50_female": "1.0,0.5",
    "100_male_75_female": "1.0,0.75",
    "100_male_100_female": "1.0,1.0",
    "100_female_75_male": "0.75,1.0",
    "100_female_50_male": "0.5,1.0",
    "100_female_25_male": "0.25,1.0",
    "100_female_0_male": "0.0,1.0"
}

TYPE_SKEWS = {
    "100_glioblastoma_0_other": "1.0,0.0",
    "100_glioblastoma_25_other": "1.0,0.25",
    "100_glioblastoma_50_other": "1.0,0.5",
    "100_glioblastoma_75_other": "1.0,0.75",
    "100_glioblastoma_100_other": "1.0,1.0",
    "100_other_75_glioblastoma": "0.75,1.0",
    "100_other_50_glioblastoma": "0.5,1.0",
    "100_other_25_glioblastoma": "0.25,1.0",
    "100_other_0_glioblastoma": "0.0,1.0"
}


def run_experiment(skew_type, skew_name, skew_values):
    experiment_config = f"configuration/unet_config_{skew_type}_{skew_name}.yaml"
    output_name = f"{BASE_OUTPUT_NAME}-{skew_name}"

    # Copy the base config file
    with open(BASE_CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)

    # Update configuration based on skew type
    config['output_name'] = output_name
    if skew_type == "age":
        config['age_skew'] = [float(x) for x in skew_values.split(",")]
        config['gender_skew'] = None
        config['ttype_skew'] = None
    elif skew_type == "gender":
        config['gender_skew'] = [float(x) for x in skew_values.split(",")]
        config['age_skew'] = None
        config['ttype_skew'] = None
    elif skew_type == "type":
        config['ttype_skew'] = [float(x) for x in skew_values.split(",")]
        config['age_skew'] = None
        config['gender_skew'] = None

    # Write updated configuration to the file
    with open(experiment_config, 'w') as file:
        yaml.safe_dump(config, file)

    print(f"Running experiment: {skew_type} skew with configuration: {skew_name}")
    print(f"Configured output_name: {output_name}")

    # Execute the experiment
    subprocess.run(["python", "train_reconstruction.py", "-c", experiment_config])

    # Optional cleanup (uncomment to delete each config file after running)
    # os.remove(experiment_config)


# Run experiments for each skew type
for skew_name, skew_values in AGE_SKEWS.items():
    run_experiment("age", skew_name, skew_values)

for skew_name, skew_values in GENDER_SKEWS.items():
    run_experiment("gender", skew_name, skew_values)

for skew_name, skew_values in TYPE_SKEWS.items():
    run_experiment("type", skew_name, skew_values)
