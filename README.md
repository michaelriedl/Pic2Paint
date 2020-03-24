# Pic2Paint
A Streamlit app for neural style transfer.

## Running the App

To run the app you will first need to setup your environment. I use Anaconda to manage my environments and have included a YAML file to recreate the environment I used. To create the environment, run the command below:
```
conda env create -n myenv -f environment.yml
```
To activate the environment you can use the command:
```
conda activate myenv
```

Once you have the environment setup you can run the Streamlit app. You can run the app with the following command:
```
streamlit run run_pic2paint.py 
```

## Processing Images

It is important to note that both the content and style image need to be the same size. If they are not, you can use the options in the sidebar to choose how to resize the images. Also, high resolution images may cause an error if using a GPU, since the processing may take up more memory than what is available on the GPU.

## Hardware Acceleration

I have written the code to utilize the system GPU if it is configured correctly. This app will still run without a GPU but will be much slower.

## Examples

Below are some examples of the style transfer output.

![](/examples/picasso.jpg)
*Style Image*

![](/examples/hard_scene.jpg)
*Content Image*

![](/examples/hard_scene_output.png)
*Output Image*

![](/examples/hard_scene_output_color.png)
*Output Image with Color Preservation*
