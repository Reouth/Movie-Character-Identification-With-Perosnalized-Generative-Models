
import os
from src import HelperFunctions
from models import CLIPmodel

def multi_image_identifier(input_dir,output_dir,image_list,model_name,device):
    # for id_imgs in os.listdir(input_dir):

  clip_pipeline = CLIPmodel.CLIPPipeline(device,model_name)
  for image_name,image,_ in image_list:
      cls = image_name.rsplit("_", 1)[0]

      df, csv_path = HelperFunctions.get_current_csv(output_dir, cls)

      image_flag = HelperFunctions.row_exist(df,cls,image_name)
      if image_flag:
          continue
      else:
          # images_path = os.path.join(input_dir,cls)
          clip_embeddings = clip_pipeline.images_to_embeddings(input_dir)
          loss = clip_pipeline.image_identifier(image,clip_embeddings)
          HelperFunctions.save_to_csv(loss,df,image_name,csv_path,reverse=True)
