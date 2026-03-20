import torch

def count_parameters_in_pt(file_path):
    print(f"Loading '{file_path}'...")
    try:
        # Load the file mapping to CPU to avoid GPU memory issues
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        total_params = 0
        
        # Case 1: The file is a dictionary (state_dict or checkpoint)
        if isinstance(data, dict):
            # Handle common training checkpoint formats
            if 'model_state_dict' in data:
                print("Detected training checkpoint. Extracting 'model_state_dict'...")
                data = data['model_state_dict']
            elif 'state_dict' in data:
                print("Detected checkpoint. Extracting 'state_dict'...")
                data = data['state_dict']
            
            # Count the parameters in the dictionary
            for key, tensor in data.items():
                if isinstance(tensor, torch.Tensor):
                    total_params += tensor.numel()
                
        # Case 2: The file is a full PyTorch model object
        elif isinstance(data, torch.nn.Module):
            print("Detected full model object...")
            total_params = sum(p.numel() for p in data.parameters())
            
        else:
            print(f"Error: Unrecognized format. Loaded object type is {type(data)}.")
            return None
            
        return total_params

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- How to use it ---
if __name__ == "__main__":
    # Replace with the path to your actual .pt or .pth file
    file_name = "bc_unet_checkpoints/best_policy.pt" 
    
    params = count_parameters_in_pt(file_name)
    
    if params is not None:
        # Prints the number with comma separators for readability
        print(f"\nTotal parameters: {params:,}")