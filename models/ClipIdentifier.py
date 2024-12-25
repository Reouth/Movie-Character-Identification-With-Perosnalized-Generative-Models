
import os
from src import HelperFunctions
from models import CLIPmodel

def multi_image_identifier(input_dir,output_dir,image_list,model_name,device):
    for id_imgs in os.listdir(input_dir):

      clip_pipeline = CLIPmodel.CLIPPipeline(device,model_name)
      df, csv_path = HelperFunctions.get_current_csv(output_dir,id_imgs)
      for image_name,image,_ in image_list:
          image_flag = HelperFunctions.row_exist(df,id_imgs,image_name)
          if image_flag:
              continue
          else:
              images_id_path = os.path.join(input_dir,id_imgs)
              clip_embeddings = clip_pipeline.images_to_embeddings(images_id_path)
              loss = clip_pipeline.image_identifier(image,clip_embeddings)
              HelperFunctions.save_to_csv(loss,df,image_name,csv_path,reverse=True)
