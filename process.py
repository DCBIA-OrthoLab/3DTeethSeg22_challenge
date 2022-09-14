import glob
from hashlib import new
import json
import os
import sys
sys.path.insert(1,'./utils')
import utils
import post_process
#import trimesh
import numpy as np
import traceback

import torch
import math


# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, PointLights,AmbientLights,TexturesVertex
)
from vtk import vtkPolyDataWriter
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import monai
from monai.transforms import ToTensor
from monai.inferers import SimpleInferer


execute_in_docker = True

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ScanSegmentation():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # use GPU if available otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("===> Using ", self.device)

        self.model_path = "checkpoints/model.pth"
        self.scal_name = "TeethSegID"
        self.nb_views = 70
        self.dist_cam = 1.35
        self.LUT = np.array([0,18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                             38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48,0])


        # create UNet
        self.num_classes = 34
        self.model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
                out_channels=self.num_classes, 
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
        )
        self.model.load_state_dict(torch.load(self.model_path,map_location=self.device))
        self.model.to(self.device)



    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        # iterate over files in input_dir, assuming only 1 file available
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(labels, instances, jaw):
        """
        Write to /output/dental-labels.json your predicted labels and instances
        Check https://grand-challenge.org/components/interfaces/outputs/
        """
        pred_output = {'id_patient': "",
                       'jaw': jaw,
                       'labels': labels,
                       'instances': instances
                       }

        if execute_in_docker:        
            with open('/output/dental-labels.json', 'w') as fp:
                json.dump(pred_output, fp, cls=NpEncoder)

        else:
            with open('./test/test_local/expected_output.json', 'w') as fp:
                json.dump(pred_output, fp, cls=NpEncoder)
        return

    @staticmethod
    def get_jaw(scan_path):
        try:
            # read jaw from filename
            _, jaw = os.path.basename(scan_path).split('.')[0].split('_')
        except:
            # read from first line in obj file
            try:
                with open(scan_path, 'r') as f:
                    jaw = f.readline()[2:-1]
                if jaw not in ["upper", "lower"]:
                    return None
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                return None

        return jaw



    @staticmethod
    def fibonacci_sphere(samples, dist_cam):
        points = []
        phi = math.pi * (3. -math.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y*y)  # radius at y
            theta = phi*i 
            x = math.cos(theta)*radius
            z = math.sin(theta)*radius
            points.append((x*dist_cam, y*dist_cam, z*dist_cam))
        return points



    def GetSurfProp(self,surf_unit):     
        surf = utils.ComputeNormals(surf_unit)

        color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))/255.0)
        verts = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(surf.GetPoints().GetData()))
        faces = ToTensor(dtype=torch.int64, device=self.device)(vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)[:,1:])
        return verts.unsqueeze(0), faces.unsqueeze(0), color_normals.unsqueeze(0)


    def predict(self, inputs):
        """
        Your algorithm goes here
        """

        try:
            assert len(inputs) == 1, f"Expected only one path in inputs, got {len(inputs)}"
        except AssertionError as e:
            raise Exception(e.args)
        scan_path = inputs[0]
        print(f"loading scan : {scan_path}")
        # read input 3D scan .obj
        try:

            SURF = utils.ReadSurf(scan_path)
            surf_unit = utils.GetUnitSurf(SURF)
            jaw = self.get_jaw(scan_path)

            print("jaw processed is:", jaw)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise



        num_faces = int(SURF.GetPolys().GetData().GetSize()/4)   
        array_faces = np.zeros((self.num_classes,num_faces))

        self.model.eval() # Switch to eval mode
        simple_inferer = SimpleInferer()


        (V, F, CN) = self.GetSurfProp(surf_unit) 
        list_sphere_points = self.fibonacci_sphere(samples=self.nb_views, dist_cam=self.dist_cam)
        list_sphere_points[0] = (0.0001, 1.35, 0.0001) # To avoid "invalid rotation matrix" error
        list_sphere_points[-1] = (0.0001, -1.35, 0.0001)


        cameras = FoVPerspectiveCameras(device=self.device)
        image_size = 320
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
                        image_size=image_size, 
                        blur_radius=0, 
                        faces_per_pixel=1, 
                        )

        lights = AmbientLights(device=self.device)
        rasterizer = MeshRasterizer(cameras=cameras,raster_settings=raster_settings)
        phong_renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(device=self.device, cameras=cameras, lights=lights))
        softmax = torch.nn.Softmax(dim=1)



        # for coords in tqdm(list_sphere_points, desc = 'Prediction      '):
        for coords in list_sphere_points:
          camera_position = ToTensor(dtype=torch.float32, device=self.device)([list(coords)])
          R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
          T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

          textures = TexturesVertex(verts_features=CN)
          meshes = Meshes(verts=V, faces=F, textures=textures)
          
          image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
          frag_object = phong_renderer.rasterizer(meshes.clone())

          depth_map = frag_object.zbuf
          pix_to_face = frag_object.pix_to_face
          image = torch.cat([image[:,:,:,0:3], depth_map], dim=-1)
          pix_to_face = pix_to_face.squeeze()
          image = image.permute(0,3,1,2)
          inputs = image.to(self.device)
          outputs = simple_inferer(inputs,self.model)  
          outputs_softmax = softmax(outputs).squeeze().detach().cpu().numpy() # t: negligeable  

          for x in range(image_size):
              for y in range (image_size): # Browse pixel by pixel
                  array_faces[:,pix_to_face[x,y]] += outputs_softmax[...,x,y]

        print("Get views: done.")

        array_faces[:,-1][0] = 0 # pixels that are background (id: 0) =-1
        faces_argmax = np.argmax(array_faces,axis=0)
        mask = 33 * (faces_argmax == 0) # 0 when face is not assigned to any pixel : we change that to the ID of the gum
        final_faces_array = faces_argmax + mask
        unique, counts  = np.unique(final_faces_array, return_counts = True)

        surf = SURF
        nb_points = surf.GetNumberOfPoints()
        polys = surf.GetPolys()
        np_connectivity = vtk_to_numpy(polys.GetConnectivityArray())




        id_points = np.full((nb_points,),33) # fill with ID 33 (gum)

        for index,uid in enumerate(final_faces_array.tolist()):
            id_points[np_connectivity[3*index]] = uid



        vtk_id = numpy_to_vtk(id_points)
        vtk_id.SetName(self.scal_name)
        surf.GetPointData().AddArray(vtk_id)

        ###
        ###
        # POST-PROCESS
        ###
        ###

        ## REMOVE ISLANDS

        # start with gum
        post_process.RemoveIslands(surf, vtk_id, 33, 500,ignore_neg1 = True) 

        for label in range(self.num_classes):
          post_process.RemoveIslands(surf, vtk_id, label, 200,ignore_neg1 = True) 


        # CLOSING OPERATION
        #one tooth at a time

        for label in range(self.num_classes):
            post_process.DilateLabel(surf, vtk_id, label, iterations=2, dilateOverTarget=False, target=None) 
            post_process.ErodeLabel(surf, vtk_id, label, iterations=2, target=None) 


        # extract UniversalID array
        labels = vtk_to_numpy(SURF.GetPointData().GetScalars(self.scal_name))



        # test jaw 
        if jaw is None:
            unique_labels = list(np.unique(labels))
            l_upper = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            l_lower = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
            c_upper =  len([i for i in unique_labels if i in l_upper])
            c_lower = len([i for i in unique_labels if i in l_lower])
            if c_upper > c_lower:
                jaw = 'upper'
            else:
                jaw = 'lower'


        # convert to their numbering system
        labels = self.LUT[labels]
        unique_point_data = np.unique (labels)
        unique_copy = np.copy(unique_point_data)
        instances = np.zeros(len(labels))

        # create instances
        for UID in unique_point_data:
            new_instance = UID
            test_new_instance = -1
            while test_new_instance != new_instance:
                test_new_instance += 1
                if test_new_instance not in (unique_copy):
                    new_instance = test_new_instance

            unique_copy[unique_copy == UID] = new_instance

            # create instance
            instances[labels == UID] = new_instance

        # extract number of vertices from mesh
        nb_vertices = len(labels)


        try:
            assert (len(labels) == len(instances)),\
                "length of output labels and output instances should be equal"
        except AssertionError as e:
            raise Exception(e.args)






        return labels, instances, jaw

    def process(self):
        """
        Read input from /input, process with your algorithm and write to /output
        assumption /input contains only 1 file
        """
        if execute_in_docker:
            # print(os.listdir('/input'))
            inputs = self.load_input(input_dir='/input/')
        else:
            print(os.listdir('./test'))
            inputs = self.load_input(input_dir='./test/')
        
        labels, instances, jaw = self.predict(inputs)
        self.write_output(labels=labels, instances=instances, jaw=jaw)


if __name__ == "__main__":
    ScanSegmentation().process()
