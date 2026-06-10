#Pete Zimmerman
#Jun-2026
#Contains: flexible network creator function
#purpose: exploratory study on MLP hyperparameter optimization
import torch
import torch.nn.functional as F

def build(params:list[int]):
    class mlp(torch.nn.Module):

        def __init__(self):
            super().__init__()
            
            layers = []
            depth = len(params)

            for i in range(0 , depth):
                if(i == 0):
                    layers.append(torch.nn.Linear(1, params[i]))
                else:
                    layers.append(torch.nn.Linear(params[i -1], params[i]))
            #output layer
            layers.append(torch.nn.Linear(params[-1], 1))
            
            #wrap in torch ModuleList
            self.layers = torch.nn.ModuleList(layers)

        #feed forward
        def forward(self, input:float):
            #pass through linear layers
            number = input

            for i in range(0 , len(self.layers) - 1):
                currLayer = self.layers[i]
                number = F.relu(currLayer(number))
            return self.layers[-1](number)


    return mlp
