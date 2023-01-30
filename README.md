# SegRec7
Classification model trained to recognize 7segmented displays in images.

Submitted for KDD class by Matej Jech, Doshisha student ID: evgh3103

Project structure:

- segrec7.py: main source of project
- segrec7_test.ipynb: visualization and testing of the model

Filepaths in the project are broken as they are taken from a different project I am working on.

The model is trained on artificialy generated images of text written in various fonts some of which are 7segmented display fonts. The model was not trained on real world data due to the lack of suitable annotated data.

Validation accuracy on generated images is 98%. Accuracy on real world data is hard to achieve due to the lack of annotated data. It is, however, definitely lower then validation on generated images.