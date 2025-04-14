import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from robofin.pointcloud.torch import FrankaSampler
from data_pipeline.geometry import construct_mixed_point_cloud

class CollisionDataset(Dataset):
    """
    Dataset for loading collision detection data with balanced classes.
    Each item contains:
    - pointcloud: combined robot, obstacle with features
    - q: robot joint configuration
    - collision_flag: binary label indicating collision (1) or free (0)
    """
    
    def __init__(self, data_file, num_robot_points=2048, num_obstacle_points=4096):
        """
        Args:
            data_file (str): Path to the pickle file containing environment data
            num_robot_points (int): Number of points to sample from robot mesh
            num_obstacle_points (int): Number of points to sample from obstacles
        """
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        
        self.fk_sampler = FrankaSampler("cpu", use_cache=True)
        
        # Load data from pickle file
        with open(data_file, 'rb') as f:
            self.all_env_data = pickle.load(f)
        
        # Process data into balanced samples
        self.samples = self._process_and_balance_data()
        
    def _process_and_balance_data(self):
        """
        Process raw environment data into balanced samples with equal numbers
        of collision and free configurations.
        
        Returns:
            List of dictionaries containing pointcloud, q, and collision_flag
        """
        # Separate free and collision samples
        free_samples = []
        collision_samples = []
        
        for env_data in tqdm(self.all_env_data, desc="Processing environments"):
            # Get obstacle point cloud (same for all samples in this environment)
            obstacle_points = construct_mixed_point_cloud(env_data["obstacles"], 
                                                           num_points=self.num_obstacle_points)
            
            # Process free configurations (collision_flag = 0)
            for q in env_data["qs_free"]:
                robot_points = self._get_robot_points(q)  
                pointcloud = self._construct_pointcloud(robot_points, obstacle_points)
                
                free_samples.append({
                    'pointcloud': pointcloud,
                    'q': torch.tensor(q, dtype=torch.float32),
                    'collision_flag': torch.tensor(0, dtype=torch.float32)
                })
            
            # Process collision configurations (collision_flag = 1)
            for q in env_data["qs_collision"]:
                robot_points = self._get_robot_points(q)
                
                pointcloud = self._construct_pointcloud(robot_points, obstacle_points)
                
                collision_samples.append({
                    'pointcloud': pointcloud,
                    'q': torch.tensor(q, dtype=torch.float32),
                    'collision_flag': torch.tensor(1, dtype=torch.float32)
                })
        
        # Balance the dataset
        min_samples = min(len(free_samples), len(collision_samples))
        balanced_samples = free_samples[:min_samples] + collision_samples[:min_samples]
        
        # Shuffle the balanced dataset
        np.random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _get_robot_points(self, q):
        """
        Generate robot surface points for configuration q.
        
        """
        q = torch.as_tensor(q).float()
        
        return self.fk_sampler.sample(q, self.num_robot_points)
    
    
    def _construct_pointcloud(self, robot_points, obstacle_points):
        """
        Construct the point cloud with features as shown in the example.
        """
        obstacle_points = torch.as_tensor(obstacle_points[:, :3]).float()
        
        xyz = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4),
                torch.ones(self.num_obstacle_points, 4),
            ),
            dim=0,
        )
        
        xyz[:self.num_robot_points, :3] = robot_points
        xyz[self.num_robot_points:self.num_robot_points+self.num_obstacle_points, :3] = obstacle_points
        
        return xyz
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def save_full_data_tensor(data_file, output_file):
    """
    Save the full dataset as a tensor file for faster loading.
    
    Args:
        data_file (str): Path to the pickle file containing environment data
        output_file (str): Path to save the tensor file
    """
    dataset = CollisionDataset(data_file)
    torch.save(dataset.samples, output_file)
    print(f"Full dataset saved to {output_file}")


def load_full_data_tensor(tensor_file):
    """
    Load the full dataset from a tensor file.
    
    Args:
        tensor_file (str): Path to the tensor file
        
    Returns:
        List of samples
    """
    samples = torch.load(tensor_file)
    print(f"Full dataset loaded from {tensor_file}")
    return samples


def get_data_loaders_from_tensor(tensor_file, batch_size=32, train_ratio=0.8, num_workers=4):
    """
    Create balanced train and validation data loaders from a pre-saved tensor file.
    
    Args:
        tensor_file (str): Path to the tensor file containing dataset samples
        batch_size (int): Batch size for data loaders
        train_ratio (float): Ratio of data to use for training (rest for validation)
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader instances for training and validation
    """
    # Load samples from tensor file
    samples = load_full_data_tensor(tensor_file)
    
    # Split indices in a stratified manner to maintain balance
    free_indices = [i for i, sample in enumerate(samples) if sample['collision_flag'] == 0]
    collision_indices = [i for i, sample in enumerate(samples) if sample['collision_flag'] == 1]
    
    # Split each class separately
    train_free_size = int(train_ratio * len(free_indices))
    train_collision_size = int(train_ratio * len(collision_indices))
    
    train_indices = free_indices[:train_free_size] + collision_indices[:train_collision_size]
    val_indices = free_indices[train_free_size:] + collision_indices[train_collision_size:]
    
    # Create subset datasets
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    DATA_FILE = "./collision_bool/raw_test.pkl"
    TENSOR_FILE = "./collision_bool/processed_test.pt"
    
    # Save full dataset as tensor
    save_full_data_tensor(DATA_FILE, TENSOR_FILE)
    
    # Create balanced data loaders from tensor
    train_loader, val_loader = get_data_loaders_from_tensor(TENSOR_FILE)
    
    # Check balance in loaders
    def check_balance(loader, name):
        total = 0
        collisions = 0
        for batch in loader:
            total += len(batch['collision_flag'])
            collisions += batch['collision_flag'].sum().item()
        print(f"{name} - Total: {total}, Collisions: {collisions}, Free: {total - collisions}")
    
    check_balance(train_loader, "Train loader")
    check_balance(val_loader, "Validation loader")
    