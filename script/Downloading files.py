""" Downloading files using Kaggle json API """


from google.colab import files 

# Select kaggle.json file
uploaded = files.upload()

# Create the .kaggle directory if it doesn't exist
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)

# Make a folder to store the data
!mkdir -p data
!kaggle datasets download -d ninadaithal/imagesoasis -p data --unzip

print("Contents of ./data:")
print(os.listdir("data"))

