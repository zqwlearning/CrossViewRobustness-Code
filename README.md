# Benchmarking the Robustness of Cross-view Geo-localization Models

To comprehensively evaluate the robustness of existing methods, this paper introduces the first benchmarks for evaluating the robustness of cross-view geo-localization models to real-world image corruptions. We applied 16 corruption types to a widely used public dataset, including CVUSA and CVACT, with 5 corruption severities per type, ultimately generating about 1.5 million corrupted images to study the robustness of different models.

Additionally, we introduce straightforward and effective robustness enhancement techniques (stylization and histogram equalization) to consistently improve the robustness of various models.

## Typical Cross-view Geo-localization models

In our work, we focus on evaluating the following typical cross-view geo-localization models.

| Method   | Paper Title                                                  | Publication | CVUSA | CVACT |
| -------- | ------------------------------------------------------------ | ----------- | ----- | ----- |
| CVM-Net  | CVM-Net: Cross-View Matching Network for Image-Based Ground-to-Aerial Geo-Localization | CVPR'18     | ✔     | ✖     |
| OriCNN   | Lending Orientation to Neural Networks for Cross-view Geo-localization | CVPR'19     | ✔     | ✔     |
| SAFA     | Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization | NeurIPS'19  | ✔     | ✔     |
| CVFT     | Optimal Feature Transport for Cross-View Image Geo-Localization | AAAI'20     | ✔     | ✔     |
| DSM      | Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching | CVPR'20     | ✔     | ✔     |
| L2LTR    | Cross-view Geo-localization with Layer-to-Layer Transformer  | NeurIPS'21  | ✔     | ✔     |
| TransGeo | TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization | CVPR'22     | ✔     | ✔     |
| GeoDTR   | Cross-view Geo-localization via Learning Disentangled Geometric Layout | AAAI'23     | ✔     | ✔     |

## Stylization

We use AdaIN (Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization) for stylized images, with style images from [painter-by-numbers](https://www.kaggle.com/c/painter-by-numbers/) and content images from ground query images in the cross-view training set.

<div style="text-align:center;">
    <image src="images/style.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
              Style images from painter-by-numbers.
        </strong>
    </p>
</div>

<div style="text-align:center;">
    <image src="images/context.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
              Content images from ground query images in the cross-view training set (CVUSA).
        </strong>
    </p>
</div>

<div style="text-align:center;">
    <image src="images/style-context.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
              Stylized ground query images.
        </strong>
    </p>
</div>

## Histogram Equalization

We use CLAHE (Contrast Limited Adaptive Histogram Equalization) for histogram-equalized images. The key code is as follows:

```python
# read images
image = cv2.imread(src_dir + grd_names[i])
b, g, r = cv2.split(image)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

clahe_b = clahe.apply(b)
clahe_g = clahe.apply(g)
clahe_r = clahe.apply(r)

enhanced_image = cv2.merge((clahe_b, clahe_g, clahe_r))
```

<div style="text-align:center;">
    <image src="images/origin.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
              Ground query images in the cross-view training set (CVUSA).
        </strong>
    </p>
</div>

<div style="text-align:center;">
    <image src="images/clahe.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
               Histogram-equalized ground query images.
        </strong>
    </p>
</div>


##Datasets / Benchmarks

See **datasets** folder for some examples.

For visualization see <a href="visualize.ipynb">visualize.ipynb</a>.

### Full Datasets / Benchmarks

Coming Soon!

<div style="text-align:center;">
    <image src="images/thumbnails.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
               Thumbnails of (some of) our benchmarks.
        </strong>
    </p>
</div>

## Corruption Robustness Evaluation

<a href="CorruptionRobustnessEvaluation.md">Corruption Robustness Evaluation</a>

### Catalog Structure of Our Datasets / Benchmarks

```shell
├─CVUSA-C
│  ├─clean
│  ├─severity-1
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  ├─severity-2
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  ├─severity-3
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  ├─severity-4
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  └─severity-5
│      ├─blur
│      │  ├─defocus_blur
│      │  ├─glass_blur
│      │  ├─motion_blur
│      │  └─zoom_blur
│      ├─digital
│      │  ├─contrast
│      │  ├─jpeg_compression
│      │  └─pixelate
│      ├─noise
│      │  ├─gaussian_noise
│      │  ├─impulse_noise
│      │  ├─shot_noise
│      │  └─speckle_noise
│      └─weather
│          ├─brightness
│          ├─fog
│          ├─frost
│          ├─snow
│          └─spatter
├─CVACT_val-C
│  ├─clean
│  ├─severity-1
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  ├─severity-2
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  ├─severity-3
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  ├─severity-4
│  │  ├─blur
│  │  │  ├─defocus_blur
│  │  │  ├─glass_blur
│  │  │  ├─motion_blur
│  │  │  └─zoom_blur
│  │  ├─digital
│  │  │  ├─contrast
│  │  │  ├─jpeg_compression
│  │  │  └─pixelate
│  │  ├─noise
│  │  │  ├─gaussian_noise
│  │  │  ├─impulse_noise
│  │  │  ├─shot_noise
│  │  │  └─speckle_noise
│  │  └─weather
│  │      ├─brightness
│  │      ├─fog
│  │      ├─frost
│  │      ├─snow
│  │      └─spatter
│  └─severity-5
│      ├─blur
│      │  ├─defocus_blur
│      │  ├─glass_blur
│      │  ├─motion_blur
│      │  └─zoom_blur
│      ├─digital
│      │  ├─contrast
│      │  ├─jpeg_compression
│      │  └─pixelate
│      ├─noise
│      │  ├─gaussian_noise
│      │  ├─impulse_noise
│      │  ├─shot_noise
│      │  └─speckle_noise
│      └─weather
│          ├─brightness
│          ├─fog
│          ├─frost
│          ├─snow
│          └─spatter
├─CVUSA-C-ALL
├─CVACT_val-C-ALL
├─CVACT_test-C-ALL
├─Stylization
└─CLAHE
```

