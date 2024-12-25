
import os
from src import HelperFunctions
from models import CLIPmodel

def multi_image_identifier(input_dir,output_dir,image_list,model_name,device):
    for image_ID in os.listdir(input_dir):

      clip_pipeline = CLIPmodel.CLIPPipeline(device,model_name)
      if HelperFunctions.image_check(image_ID):
        cls = image_ID.rsplit(".", 1)[0]
      else:
          print("file {} not an image".format(image_ID))
          break
      df, csv_path = HelperFunctions.get_current_csv(output_dir,cls)
      for image_name,image,_ in image_list:
          image_flag = HelperFunctions.row_exist(df,cls,image_name)
          if image_flag:
              continue
          else:
              image_id_path = os.path.join(input_dir,image_ID)
              clip_embeddings = clip_pipeline.images_to_embeddings(image_id_path)
              loss = clip_pipeline.image_identifier(image,clip_embeddings)
              HelperFunctions.save_to_csv(loss,df,image_name,csv_path,reverse=True)
