from PIL import Image, ImageDraw, ImageFont
import os
import random
from tqdm import tqdm
import shutil
import config as config
random.seed(42)
class PrepareDataset:
    def __init__(self, project_files_path):
        """
        Initializes the PrepareDataset class.

        Args:
            dataset_path (str): The path to the dataset.

        Attributes:
            project_files_path (str): The path to the project files.
            dataset_path (str): The path to the dataset.
            text_file_path (str): text file containing 10,000 preprocessed most common English words.
            words (list): A list of preprocessed English words.
            train_ratio (float): The ratio of data to be used for training.
            test_ratio (float): The ratio of data to be used for testing.
            validation_ratio (float): The ratio of data to be used for validation.
        """
        self.project_files_path = project_files_path
        self.font_data_path = os.path.join(self.project_files_path, "data")
        self.text_file_path=os.path.join(self.project_files_path, "google-10000-english-preprocessed.txt")
        with open(self.text_file_path, 'r') as file:
            self.words = file.read().splitlines()
        self.train_ratio = 0.8
        self.test_ratio = 0.1
        self.validation_ratio = 0.1
        #name of the folder where the synthetic data will be stored
        #synthetic_data_20_5 means 20 samples per image for training and 5 samples per image for validation
        self.synthetic_folder_name=f'synthetic_data_{config.num_samples_per_image_train}_{config.num_samples_per_iamge_val}'
        #write the folder name to the config file
        with open('config.py','a') as file:
            file.write(f'\nsynthetic_folder_name="{self.synthetic_folder_name}"\n')
        file.close()
        print('Initialized PrepareDataset class')

    
    def get_folder_names(self):
        """
        Returns the names of the font folders in the font_data_path
        """
        if(not os.path.exists(self.font_data_path)):
            print("Font Data folder does not exist. Please specify correct path to the font data folder.")
            return
        folder_names = []
        for root, dirs, files in os.walk(self.font_data_path):
            for dir in dirs:
                folder_names.append(dir)
        return folder_names
    
    def get_num_chars(self,width):
        """
        Returns the number of characters to be generated in the image based on the width of the image.
        """
        num_chars=(width//config.width_char_ratio)+random.randint(-config.offset_char,config.offset_char)
        return max(5,num_chars)
    
    def generate_string(self,num_chars):
        """
        Returns a string of words with the total number of characters equal to or less than num_chars.
        Words are randomly selected from the list of preprocessed English words. 
        """
        result = ''
        random.shuffle(self.words) 
        while len(result) < num_chars:
                random_word = random.choice(self.words)
                if len(result) + len(random_word) <= num_chars:
                    result += random_word + ' '
                else:
                    if abs(len(result) + len(random_word) - num_chars) < 3:
                        break
                    else:
                        random_word = random.choice([word for word in self.words if len(word) <= (num_chars - len(result))])
                        result += random_word + ' '     
        return result.strip()


    def splitting_data(self):
        """
        Splits the data into train, test and validation sets for each font.
        """
        font_names_list=self.get_folder_names()
        print('Total number of fonts in dataset:',len(font_names_list))
        
        #create folders for train, test and val data for each font
        for font_name in font_names_list:
            os.makedirs(os.path.join(self.project_files_path,self.synthetic_folder_name,"train",font_name),exist_ok=True)
            os.makedirs(os.path.join(self.project_files_path,self.synthetic_folder_name,"test",font_name),exist_ok=True)
            os.makedirs(os.path.join(self.project_files_path,self.synthetic_folder_name,"val",font_name),exist_ok=True)
            #get all the images in the font folder
            images=os.listdir(os.path.join(self.project_files_path,"data",font_name))
            random.shuffle(images)
            train_images=images[:int(len(images)*self.train_ratio)]
            test_images=images[int(len(images)*self.train_ratio):int(len(images)*(self.train_ratio+self.test_ratio))]
            val_images=images[int(len(images)*(self.train_ratio+self.test_ratio)):]
            #check there are no common images in the train, test and val sets
            assert len(set(train_images).intersection(set(test_images)))==0
            assert len(set(train_images).intersection(set(val_images)))==0
            assert len(set(test_images).intersection(set(val_images)))==0

            #copy the images to the respective folders
            for image in train_images:
                shutil.copy(os.path.join(self.project_files_path,"data",font_name,image),os.path.join(self.project_files_path,self.synthetic_folder_name,"train",font_name,image))
            for image in test_images:
                shutil.copy(os.path.join(self.project_files_path,"data",font_name,image),os.path.join(self.project_files_path,self.synthetic_folder_name,"test",font_name,image))
            for image in val_images:
                shutil.copy(os.path.join(self.project_files_path,"data",font_name,image),os.path.join(self.project_files_path,self.synthetic_folder_name,"val",font_name,image))
            print(f"Created folders for font {font_name} with  {len(train_images)} train images {len(val_images)}  val iamges and  {len(test_images)} test iamges " )
        print('-'*50)
    
    def generate_sample_font_images(self,font_name):
        """
        Generates synthetic images for each font in the dataset for training and validation
        """
        os.makedirs(os.path.join(self.project_files_path,self.synthetic_folder_name,"synthetic_train",font_name),exist_ok=True)
        os.makedirs(os.path.join(self.project_files_path,self.synthetic_folder_name,"synthetic_val",font_name),exist_ok=True)
        font_data_folder_train=os.path.join(self.project_files_path,self.synthetic_folder_name,"train",font_name)
        font_data_folder_val=os.path.join(self.project_files_path,self.synthetic_folder_name,"val",font_name)
        #using the ttf file for generating images
        font_path = os.path.join(self.project_files_path,"fonts",font_name+'.ttf')
        
        print(f'Using {font_path} for generating images.')
        font_size=72
        
        font_files_train=[os.path.join(font_data_folder_train,font_file) for font_file in os.listdir(font_data_folder_train)]
        font_files_val=[os.path.join(font_data_folder_val,font_file) for font_file in os.listdir(font_data_folder_val)]
        
        files={'train':font_files_train,'val':font_files_val}
        num_files={'train':config.num_samples_per_image_train,'val':config.num_samples_per_iamge_val}
        new_files_generated=0
        for key in files.keys():
            for font_file in tqdm((files[key])):
                #print('Processing:',font_file)
                img = Image.open(font_file)
                width, height = img.size
                #save the cuurent file
                if(key=='train'):
                    new_path=font_file.replace('train','synthetic_train')
                else:
                    new_path=font_file.replace('val','synthetic_val')
                #print('Saving to:',new_path)
                img.save(new_path)
                
                #generate new images
                for i in range(num_files[key]):
                    font = ImageFont.truetype(font_path, font_size)
                    num_chars=self.get_num_chars(width)
                    #generate the text
                    text = self.generate_string(num_chars)
                    #draw the text on the image
                    image = Image.new("RGB", (width, height), "white")
                    draw = ImageDraw.Draw(image)
                    text_length = draw.textlength(text, font=font)
                    #reduce the font size if the text is too long``
                    while text_length > width:
                        font_size -= 1
                        font = ImageFont.truetype(font_path, font_size)
                        text_length = draw.textlength(text, font=font)
                    font_size+=1
                    font = ImageFont.truetype(font_path, font_size)
                    text_position = (image.width//2, image.height//2 )
                    draw.text(text_position, text, fill="black", font=font,anchor="mm")
                    try:
                        image.save(new_path.replace('.png',f'_{i}.png'))
                        new_files_generated+=1
                    except:
                        print('Error saving image for')
                        print(font_file)
        print(f'Generated {new_files_generated} new images for {font_name}')
        print('-'*50)

if __name__ == "__main__":
    #define the path of the project files folder
    project_files_path = os.path.join(os.getcwd(), "project_files")
   
    prepare_dataset = PrepareDataset(project_files_path)
    #split the data into train, test and val sets
    prepare_dataset.splitting_data()
    #generate synthetic images for each font
    font_names_list=prepare_dataset.get_folder_names()
    for font_name in font_names_list:
        prepare_dataset.generate_sample_font_images(font_name)
    folder_name=f'synthetic_data_{config.num_samples_per_image_train}_{config.num_samples_per_iamge_val}'
    path = os.path.join(project_files_path,folder_name)
    print(f'Successfully generated synthetic dataset in {path} folder.')
    