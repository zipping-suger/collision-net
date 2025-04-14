import os
import pickle
from tqdm import tqdm  
from environments.cubby_environment import CubbyEnvironment
from robofin.collision import FrankaSelfCollisionChecker
 

def generate_data(output_file, num_envs=10, samples_per_env=50, margin=0):
    """
    Generate diverse environments and sample configurations (q) and poses.

    Args:
        output_file (str): File to save the generated data.
        num_envs (int): Number of environments to generate.
        samples_per_env (int): Number of samples (q, pose) per environment.
        margin (float): Margin parameter for sampling.
    """
    # Initialize a new environment
    env = CubbyEnvironment()  
    selfcc = FrankaSelfCollisionChecker()
    
    all_env_data = []  # List to store data for all environments

    # Use tqdm to show progress
    for env_id in tqdm(range(num_envs), desc="Generating Environments"):
        # Sample q and pose data
        obstacles, qs_free, poses_free, qs_collision, poses_collision = env.sample_q_pose(
            selfcc=selfcc, how_many=samples_per_env, margin=margin
        )

        # Collect the data for this environment
        env_data = {
            "env_id": env_id,
            "obstacles": obstacles,
            "qs_free": qs_free,
            "poses_free": poses_free,
            "qs_collision": qs_collision,
            "poses_collision": poses_collision,
        }

        all_env_data.append(env_data)  # Append to the list
        
    # ceate the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save all environment data to a single pickle file
    with open(output_file, "wb") as f:
        pickle.dump(all_env_data, f)

    print(f"All environment data saved to {output_file}")


if __name__ == "__main__":
    # Customize these parameters as needed
    OUTPUT_FILE = "./collision_bool/raw_test.pkl"
    NUM_ENVS = 1_00
    SAMPLES_PER_ENV = 5
    MARGIN = 0
   
    generate_data(OUTPUT_FILE, NUM_ENVS, SAMPLES_PER_ENV, MARGIN)