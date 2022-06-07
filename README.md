# DefConvs_HTR
Boosting modern and historical handwritten text recognition with deformable convolutions (ICPR20, IJDAR)

![ezgif com-gif-maker](https://user-images.githubusercontent.com/11275056/172329689-e7abc318-51de-44ff-bcb1-a8d2d225eda8.gif)
<p><img src="https://user-images.githubusercontent.com/11275056/172329689-e7abc318-51de-44ff-bcb1-a8d2d225eda8.gif"></p>

This repo contains the source code for the papers:
- Cojocaru, Iulian, et al. "Watch your strokes: Improving handwritten text recognition with deformable convolutions." 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021.
- Cascianelli, Silvia, et al. "Boosting modern and historical handwritten text recognition with deformable convolutions." International Journal on Document Analysis and Recognition (IJDAR) (2022): 1-11.

## Abstract
Handwritten Text Recognition (HTR) in free-layout pages is a challenging image understanding task that can provide a relevant boost to the digitization of handwritten documents and reuse of their content. The task becomes even more challenging when dealing with historical documents due to the variability of the writing style and degradation of the page quality. 

State-of-the-art HTR approaches usually encode input images with Convolutional Neural Networks, whose kernels are typically defined on a fixed grid and focus on all input pixels independently. Since convolutional kernels are defined on fixed grids and focus on all input pixels independently while moving over the input image, this strategy, this is in contrast with the sparse nature of handwritten pages, in which only pixels representing the ink of the writing are useful for the recognition task. Furthermore, the standard convolution operator is not explicitly designed to take into account the great variability in shape, scale, and orientation of handwritten characters. 

To cope with these specific HTR difficulties, we propose to adopt deformable convolutions, which can deform depending on the input at hand and better adapt to the geometric variations of the text. 

<p dir="auto">The considered architectures are inspired by those presented in <em><a href="http://www.jpuigcerver.net/pubs/jpuigcerver_icdar2017.pdf" rel="nofollow">Are Multidimensional Recurrent Layers Really Necessary for Handwritten Text Recognition?</a></em> (1D-LSTM) and <em><a href="https://arxiv.org/pdf/1507.05717.pdf" rel="nofollow">An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition.</a></em>.</p>

## Dependencies
Please refer to the environment.yml file

## References
If you find this repo useful, please consider citing:

<div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>@article{cascianelli2022boosting,
  title={Boosting modern and historical handwritten text recognition with deformable convolutions},
  author={Cascianelli, Silvia and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  pages={1--11},
  year={2022},
  publisher={Springer}
}
</code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@article{cascianelli2022boosting,
  title={Boosting modern and historical handwritten text recognition with deformable convolutions},
  author={Cascianelli, Silvia and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  pages={1--11},
  year={2022},
  publisher={Springer}
}" tabindex="0" role="button" style="display: inherit;">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
  
  <div class="snippet-clipboard-content notranslate position-relative overflow-auto"><pre class="notranslate"><code>@inproceedings{cojocaru2021watch,
  title={Watch your strokes: Improving handwritten text recognition with deformable convolutions},
  author={Cojocaru, Iulian and Cascianelli, Silvia and Baraldi, Lorenzo and Corsini, Massimiliano and Cucchiara, Rita},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={6096--6103},
  year={2021},
  organization={IEEE}
}
</code></pre><div class="zeroclipboard-container position-absolute right-0 top-0">
    <clipboard-copy aria-label="Copy" class="ClipboardButton btn js-clipboard-copy m-2 p-0 tooltipped-no-delay" data-copy-feedback="Copied!" data-tooltip-direction="w" value="@inproceedings{cojocaru2021watch,
  title={Watch your strokes: Improving handwritten text recognition with deformable convolutions},
  author={Cojocaru, Iulian and Cascianelli, Silvia and Baraldi, Lorenzo and Corsini, Massimiliano and Cucchiara, Rita},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={6096--6103},
  year={2021},
  organization={IEEE}
}" tabindex="0" role="button" style="display: inherit;">
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2">
    <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
</svg>
      <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2">
    <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
</svg>
    </clipboard-copy>
  </div></div>
