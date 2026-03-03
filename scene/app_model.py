import torch
import torch.nn as nn
import os

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class AppModel(nn.Module):
    def __init__(self, num_images=1600):  
        super().__init__()   
        self.appear_ab = nn.Parameter(torch.zeros(num_images, 2).cuda())
        self.optimizer = torch.optim.Adam([
                                {'params': self.appear_ab, 'lr': 0.001, "name": "appear_ab"},
                                ], betas=(0.9, 0.99))
            
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "app_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"save app model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'app.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "app_model"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "app_model/iteration_{}/app.pth".format(loaded_iter))
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)
