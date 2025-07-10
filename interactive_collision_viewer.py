import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robofin.pointcloud.torch import FrankaSampler
from robofin.collision import FrankaSelfCollisionChecker
from data_pipeline.geometry import construct_mixed_point_cloud
from data_pipeline.environments.cubby_environment import CubbyEnvironment
from train import CollisionNetPL

def check_collision(model, fk_sampler, q, obstacle_points, num_robot_points, num_obstacle_points, device):

    q_tensor = torch.as_tensor(q).float().to(device)   
    # Time the robot point sampling operation
    sample_start = time.time()
    robot_points = fk_sampler.sample(q_tensor.cpu().unsqueeze(0), num_robot_points)
    robot_points_np = robot_points.squeeze(0).cpu().numpy()
    sample_end = time.time()
    print(f"Robot point sampling time: {(sample_end - sample_start)*1000:.2f}ms")

    obstacle_tensor = torch.as_tensor(obstacle_points[:, :3]).float().to(device)
    
    # xyz = torch.cat((
    #     torch.zeros(num_robot_points, 4).to(device),
    #     torch.ones(num_obstacle_points, 4).to(device),
    # ), dim=0)
    # xyz[:num_robot_points, :3] = torch.as_tensor(robot_points_np).to(device)
    # xyz[num_robot_points:, :3] = obstacle_tensor
    
    xyz = torch.ones(num_obstacle_points, 4).to(device)
    xyz[:, :3] = obstacle_tensor

    # Time the model inference operation
    inference_start = time.time()
    with torch.no_grad():
        pred = model(xyz.unsqueeze(0), q_tensor.unsqueeze(0))
        print(f"Model output: {pred}")
        collision = pred.item() > 0.5
    inference_end = time.time()
    print(f"Model inference time: {(inference_end - inference_start)*1000:.2f}ms")

    return collision, robot_points_np

def main():
    env = CubbyEnvironment()
    selfcc = FrankaSelfCollisionChecker()
    # Check if CUDA is available and set device accordingly Point Transformer V3 only supports GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")  # Force CPU to test if crash is CUDA-related

    
    fk_sampler = FrankaSampler("cpu", use_cache=True)

    checkpoint_path = 'checkpoints_ptv_nr_256/collisionnet-epoch=36-val_loss=0.10.ckpt'
    model = CollisionNetPL.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device).eval()
    # model = CollisionNetPL().to(device).eval()
    
    # Print model prarameters count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    num_robot_points = 2048
    num_obstacle_points = 4096

    obstacles, q_list_free, _, _, _ = env.sample_q_pose(selfcc=selfcc, how_many=1, margin=0)
    obstacle_points = construct_mixed_point_cloud(obstacles, num_points=num_obstacle_points)
    q = q_list_free[0]

    # Safe key mappings: joint 1 → z/x, joint 2 → c/v, ..., joint 7 → p/;
    key_to_joint_delta = {
        ']': (0, +0.1), '[': (0, -0.1),
        'p': (1, +0.1), 'o': (1, -0.1),
        'i': (2, +0.1), 'u': (2, -0.1),
        ';': (3, +0.1), 'l': (3, -0.1),
        'k': (4, +0.1), 'j': (4, -0.1),
        'h': (5, +0.1), 'g': (5, -0.1),
        'm': (6, +0.1), 'n': (6, -0.1)
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    robot_scatter = ax.scatter([], [], [], c='r', s=1, label='Robot')
    obstacle_scatter = ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2],
                                  c='g', s=1, label='Obstacle')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot and Obstacle Point Clouds')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

    def update_visualization():
        collision, robot_points_np = check_collision(
            model, fk_sampler, q, obstacle_points, num_robot_points, num_obstacle_points, device)

        # Update robot scatter points
        robot_scatter._offsets3d = (robot_points_np[:, 0], robot_points_np[:, 1], robot_points_np[:, 2])

        ax.set_title(f"Robot and Obstacle Point Clouds — Collision: {'YES' if collision else 'NO'}")
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        if key in key_to_joint_delta:
            joint_idx, delta = key_to_joint_delta[key]
            q[joint_idx] += delta
            print(f"Joint {joint_idx+1} {'+' if delta > 0 else '-'}0.1 rad → {q[joint_idx]:.3f}")
            update_visualization()
        elif key == 'escape':
            print("Exiting.")
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_visualization()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
