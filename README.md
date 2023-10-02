# Satellite_Imagery_SemanticSegmentation
## Motivation
The primary objective of this project revolves around tackling the complexities associated with training deep learning algorithms on large images, particularly when facing resource constraints. The ultimate goal is to gain a comprehensive understanding of effective strategies for handling these challenges. By delving into this endeavor, I aim to shed light on the performance analysis of diverse deep learning algorithms when trained on processed large-scale images.Working with large images in training deep learning models for semantic segmentation poses several challenges. Firstly, the memory demands of these images often surpass the available GPU memory capacity, making it difficult to load and process them entirely.  Secondly, the computational workload associated with forward and backward passes through a deep neural network with large images is high. Training on such data can extend the model development process. Lastly, memory constraints often lead to the reduction of batch sizes during training. Smaller batches are used to fit within the available memory, but this reduction can result in less stable training dynamics and slower convergence rates, potentially impeding the learning process. 

## Data Discription
The dataset comprises a collection of images drawn from aerial photographs used in the creation of a comprehensive digital orthophoto encompassing the entire region of Poland. These images originate from publicly available geodetic resources. They were captured at spatial resolutions of either 25 or 50 centimeters per pixel and consist of three spectral bands in RGB format. The dataset encompasses images exhibiting varying levels of saturation, sunlight angles, and shadow lengths, capturing diverse environmental conditions throughout different stages of the vegetation season. To ensure maximum diversity, the dataset's creators manually handpicked 41 orthophoto tiles from different counties spanning all regions, including 33 images at a resolution of 9000 × 9500 pixels and 8 images at a resolution of 4200 × 4700 pixels.

<table>
  <tr>
    <td align="center">
      <img src="image1.jpg" alt="Image 1" width="300">
      
    </td>
    <td align="center">
      <img src="image2.jpg" alt="Image 2" width="300">
    </td>
    <br>
  </tr>
</table>




