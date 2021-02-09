# Table2Charts: Recommending Charts by Learning Shared Table Representations
This repository provides code and data of the paper _Table2Charts: Recommending Charts by Learning Shared Table Representations_.

## Table2Charts Code
The core parts included in the folder [`Table2Charts`](Table2Charts) are:
* The PyTorch code of the deep Q-network with copying mechanism mentioned in Figure 4 and section 3.1 of the paper. The major details are in [`Table2Charts/DQN with Copying/copyNet.py`](Table2Charts/DQN%20with%20Copying/copyNet.py).
* The Python code of the heuristic beam searching agent mentioned in section 3.2 and appendix A.3 of the paper. The major details are in [`Table2Charts/Heuristic Beam Searching/drill_down.py`](Table2Charts/Heuristic%20Beam%20Searching/drill_down.py).

## Baselines Code
In the paper Table2Charts is compared with the following four baselines:
* DeepEye: From the paper _DeepEye: Towards Automatic Data Visualization_ with inference models at <https://github.com/Thanksyy/DeepEye-APIs>.
* Data2Vis: From the paper _Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks_ with code at <https://github.com/victordibia/data2vis>.
* VizML: From the paper _VizML: A Machine Learning Approach to Visualization Recommendation_ with code and data at <https://github.com/mitmedialab/vizml>.
* DracoLearn: From the paper _Formalizing Visualization Design Knowledge as Constraints: Actionable and Extensible Models in Draco_ with inference models at <https://github.com/uwdata/draco>.

In the folder [`Baselines`](Baselines), we provide more details on how we train and evaluate those baselines.

## Data
In addition to our Excel chart corpus (which is under privacy reviews for publication), we use two more datasets for comparing with baselines in section 4.2:
* A public Plotly corpus used in VizML paper.
* 500 HTML tables (crawled from the public web) for human evaluation.

In the folder [`Data`](Data), we provide the way we get and process Plotly corpus, and the results about human evaluation.
