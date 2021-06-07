import numpy as np
import config_DWRegal as configFile
import numpy as np
import torch


class GUI_config_fingerprinting_table:
    #def __init__(self):
        #self.load_path=load_path

    def main(self,path):
        app = configFile.App
        numberOfCores = configFile.numberOfCores
        ## Header information:
        author = configFile.Author
        additional_info = 'Origin of the inertial frame: x=0, y=0, z=0'
        ## Table format: (table entries will be depicted in given format)
        # exponential representation with 5 decimal places
        tableFormat = '%10.5E'
        ## Select a set of Positions and rotations or add a new one:

        tableName = configFile.tableName

        positions_tm = torch.load(path)
        positions = []
        for i in range(len(positions_tm)):  # 355,385): #591,611):#
            if (np.isnan(positions_tm[i, 0]) == False):
                positions.append(positions_tm[i, :])
        # positions = np.array(positions)
        # positions =positions[[2000],:]
        # positions = [xpos, ypos, zpos]

        ## 2. define rotation angles
        unit = 'deg'

        ## rotation angles
        angles = [[0], [0], [0], unit]
        # print(angles)
        return app,numberOfCores,author,tableFormat,positions,angles,additional_info
#if __name__ == '__main__':
#    gcft = GUI_config_fingerprinting_table()
#    gcft.main()


