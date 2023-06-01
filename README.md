# Using Machine Learning to Predict Alcohol and Drug Use with Personality Traits and Socio-Demographic Characteristics 

_EXAMINING HEALTH OUTCOMES USING MACHINE LEARNGING_


## Table of Contents
* [Purpose of Project](#purpose-of-project)
* [Project Summary](#project_summary)
* [List of key files](#list_of_key_files)
* [Python Libraries](#python_libraries)
* [Data Description](#data-descrption)
* [Methodology](#methodology)
* [Results](#results)
* [Conclusion](#conclusion)
* [Creator & Credits](#creators)


## Purpose of Project

The current project explores health outcomes and machine learning for my Udacity Data Scientist Nanodegree Capstone project. The overall goal was to explore if machine learning models could help better predict drug use from personality and socio-demographic variables.

This repository will contain the literature review, code, analysis, and visualizations. Three classes of drugs are examined: **stimulants, depressants, and hallucinogens**. Machine-learning models were applied to predict the consumption of each class of drug based on the risk factors.

For the full write up, please see my **three-part blog** on Medium.com, The links are the following:

For **Part 1** (A brief literature review on drug use, personality and previous methods used to study the associations between the two, the study methodology, explanation of the methods used and chosen metrics): Use this [Link](https://medium.com/@andrew.sleung/using-machine-learning-to-predict-alcohol-and-drug-use-with-personality-traits-and-8ec09dc16058).
For **Part 2** (The project's exploratory analysis, Data-Cleaning/Pre-Processing phase, Feature Selection and Implementing the machine learning models): Use this [Link](https://medium.com/@andrew.sleung/using-machine-learning-to-predict-alcohol-and-drug-use-with-personality-traits-and-b6ea74eec3e7).
For **Part 3** (The project's results, discussing how the research questions were answered and concludes with looking at possible future improvements): Use this [Link](https://medium.com/@andrew.sleung/using-machine-learning-to-predict-alcohol-and-drug-use-with-personality-traits-and-694d985ad114).



## Project Summary <a name="project_summary"></a>

_**Introduction:**_

Alcohol consumption, illicit drug use and non-illicit drug use represent a major source of mortality and morbidity, as well as social and economic costs in Canada(Canada, 2019; Thomas & Davis, 2007). Studies have highlighted the complex relationship between drug use and a range of factors, including psychological, geographical, environmental, socio-economic, and individual risk factors (Cleveland, Feinberg, Bontempo, & Greenberg, 2008; Ventura, de Souza, Hayashida, & Ferreira, 2015). Improving our understanding of how these risk factors contribute to alcohol and drug use can create more effective individual-level treatments and help health agencies craft policies and programs that provide harm-reduction and enable prevention-based approaches. 

_**Dataset and Research Questions:**_

The current data set includes 1885 respondents and their usage of eighteen different illicit and non-illicit drugs (Dua & Graff, 2019). These eighteen different drugs can be groupd into three major classes of substances: Stimulants, Depressants and Hallucinogens.

 Respondent demographics and five dimensions of personality following the five-factor model (Terracciano, Löckenhoff, Crum, Bienvenu, & Costa, 2008) are also included in the dataset which provide the opportunity to make use of machine-learning approaches to better understand the association between socio-demographic and personality traits with alcohol and drug use. The following two research questions are addressed:

1)	What are the personality traits and demographic variables that best predict each drug use outcome? 

2)	Can we determine which machine-learning approaches/methods is the most effective for predicting consumption? 

In the exploratory analysis and data mining phase of the project, there will be additional opportunities to unearth new questions that we can also answer.

_**Methods & Analysis:**_

To obtain a detailed understanding of the respondents, exploratory analysis was performed, including descriptive statistics, visualizations, bivariate regressions and data mining. A preliminary analysis of the data showed that drug use is captured in frequency categories, indicating that a classification approach would be most appropriate in analyzing the data. There was also the potential to group drug use labels into binary outcomes or leave them as multi-class problems. Following exploratory analysis, to improve the sample size and reduce the complexity of the problem the eighteen drug outcomes were collapsed into three categories: stimulants, depressants, and hallucinogens. The frequency class labels for the drug outcomes were also collapsed from seven labels to three (non-user, infrequent user, and high use). 

Following data preparation, initial models were created using the following algorithms:

-	Logistic Regression
-	Multinomial Logit Regression (Softmax Regression)
-	Support Vector Machines
-	Decision Trees/Random Forests/Gradient Boosted Trees
-	K-Nearest Neighbours classifier
-	Linear Discriminant Analysis (LDA)

After further modelling, four models were selected:

+ Logistic Regression
+ Support Vector Machines
+ Random Forests
+ Neural Network

Imbalanced data was then fixed with oversampling using SMOTE before the models were then re-calibrated. A third binarized approach was then used to nest the four classifiers into the OneVsRest classifier and re-calibrate the models to allow the AUC evaluation metric to be used.

Finally, the best set of hyperparameters and features were searched for using GridSerachCV and applied to each of the models for one last calibration before examining the evaluation metrics to determine the best model. Feature importance from the random forest classifier and coefficients from multinomial logistic regressions were used to assess significant predictors of drug use outcomes.


_**Tools and Techniques:**_

Python will be used as the primary tool for this project. Data preparation, exploratory analysis and data modelling code will be saved in Jupyter notebooks or .py files.


_**Evaluation Metrics:**_

To assess the performance of the models in predicting drug use from the risk factors, the following metrics will be calculated:

1.	Accuracy
2.	Precision/Specificity
3.	Recall/Sensitivity
4.	AUC


## File structure and list of key files <a name="list_of_key_files"></a>

* **01_Exploratory_Analysis_Modules**
    * 01a_EDA_Descriptives_and_Visualizations_20230510.ipynb                   #descriptive statistics and visualizations - notebook
    * 01b_EDA_Correlations_and_Initial_Models_20230511.ipynb                   #correlations and preliminary models - notebook

* **02_Data_Pre-Processing_and_Initial_Models**
    * 02a_Data_Preprocessing_and_Addtional_Data_Exploration_20230512.ipynb     #data pre-processing and clean script with additional EDA
    * 02b_Data_Preprocessing_and_Initial_Models_Full_Run_20230513.ipynb        #creating initial model scipts using Stimulants outcome to prototype
    * 02c_Testing_rebalanced_models_and_binarize_20230514.ipynb                #testing re-balanced models and a OneVsRest/Binarized approach
    * 02d_Developing_Model_Optimization_Testing_20230515.ipynb                 #developing the paramater grid for GridSearchCV

* **03_Modelling_Cross_Validation_Optimization_Stimulants**
    * 03a_End_to_End_ML_Modelling_for_Stimulants_20230516.ipynb                #End-to-end pipeline script to predict Stimulants usage with ML models
    * 03b_Hyperparameter_Optimization_for_Stimulants_Models_20230517.ipynb     #GridSerachCV for the best parameters for ML models used to predict Stimulants usage

* **04_Modelling_Cross_Validation_Optimization_Depressants**
    * 04a_End_to_End_ML_Modelling_for_Depressants_20230518.ipynb               #End-to-end pipeline script to predict Depressants usage with ML models
    * 04b_Hyperparameter_Optimization_for_Depressants_Models_20230519.ipynb    #GridSerachCV for the best parameters for ML models used to predict Depressants usage

* **05_Modelling_Cross_Validation_Optimization_Hallucinogens**
    * 05a_End_to_End_ML_Modelling_for_Hallucinogens_20230520.ipynb             #End-to-end pipeline script to predict Hallucinogens usage with ML models
    * 05b_Hyperparameter_Optimization_for_Hallucinogens_Models_20230521.ipynb  #GridSerachCV for the best parameters for ML models used to predict Hallucinogens usage

* **visuals**
	* data_description_1.png     #descriptive statistics for independent variables
	* data_description_2.png     #output classes for all drug outcome variables
	* data_description_3.png  	 #distribution of values for outcome features 	
	* methodology_20230530.png   #methodology diagram


* drug_consumption_cap_20230505.csv #input dataset - drug use along with personality traits and socio-demographic variables
* **README.md**

## Python_Libraries

* nltk
* sklearn
* joblib
* sqlqlchemy
* time
* numpy
* re
* pandas
* scipy
* seaborn
* matplotlib
* statsmodels

## Data Description

The dataset for the study was obtained from the from the open source repository - UCI Machine Learning Repository. See the following link: 

http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29#

Alternatively, a cleaned version can also be found in this repository:

https://github.com/as2leung/predicting_drug_use_from_personality_traits_using_ML/blob/master/drug_consumption_cap_20230505.csv

### Independent variables

In this drug consumption dataset, there are five sociodemographic variables (Age, Gender, Education, Country, and Ethnicity), seven measures of personality (NEO-FFI-R/FFM: Neuroticism, Extraversion, Openness, Agreeableness, Conscientiousness; Impulsiveness, Sensation Seeking) and one row identifier (ID). Only ID will not be used as an input feature as it provides nothing more than a unique row count. The table below provides descriptive statistics for the 12 + 1 (ID) independent variables.

![Independent_Descriptives](https://github.com/as2leung/predicting_drug_use_from_personality_traits_using_ML/blob/master/visuals/data_description_1.png)

### Outcome variables
The data set contains eighteen outcome variables of drug use that are all categorical and are labelled with the same output classes. The table below outlines all the class labels and the respective definitions. One potential transformation might be having to group up the classes into only two or 3 classes depending on how imbalanced the classes are (see Data Preparation Transformation Column)

![Outcome_Classes](https://github.com/as2leung/predicting_drug_use_from_personality_traits_using_ML/blob/master/visuals/data_description_2.png)

For this study, the eighteen different drugs will also be grouped into three broader classes of drugs. The groupings will be:

<b>Data Preparation – Outcome variables Transformation  </b>
| Major Class of Drug | Drug |
| :--- | :----: |
| Stimulants  | amphetamines, nicotine, cocaine powder, crack cocaine, caffeine, and chocolate |
| Depressants | alcohol, amyl nitrite, benzodiazepines, tranquilizers, solvents and inhalants, heroin and methadone/prescribed opiates | 
| Hallucinogens | cannabis, ecstasy, ketamine, LSD, and magic mushrooms |

The original distributions for eighteen drugs are as follows:

![Outcome_Distribution](https://github.com/as2leung/predicting_drug_use_from_personality_traits_using_ML/blob/master/visuals/data_description_3.png)

## Methodology

The methodology for the current study is presented in the following diagram:

![Methodology_Flowchart](https://github.com/as2leung/predicting_drug_use_from_personality_traits_using_ML/blob/master/visuals/methodology_20230530.png)


## Results

In carrying out the study the neural network was determined to be the most accurate algorithm in predicting stimulant drug use via F1-score. For depressants and hallucinogens, the random forest algorithm was able to produce the best estimates with both drug outcomes. While the findings are not all significant and caution needs to be taken, this study was able to demonstrate that traits like neuroticism, agreeableness, impulsiveness, and sensation seeking behaviour were predictive of drug use outcomes, along with other sociodemographic variables.

The first section of results tables summarizes the feature imporance for each drug use outcome to answer rsearch question one. The second section of tables provides the model metrics for each of the four models used by drug use outcome. A best model is indicated for each drug use outcome, with the overall goal of using these metrics to maximize the accuracy score of the predictions, while balancing precision and recall scores.

### Results for Research Question One - Predictors of Drug Use

<b>Top Features by Feature Importance - Stimulants</b>
|Feature|Impurity-based Importance|
|:----:|:----:|
NEO_Neuroticism|0.212085|
|Ethnicity|0.195933|
|Gender|0.143130|


<b>Top Features by Feature Importance - Depressants</b>
| Feature           | Impurity-based Importance   |
|:-------------------:|:-----------------------------:|
| Ethnicity         | 0.431104                    |
| Sensation Seeking | 0.264706                    |
| Impulsiveness     | 0.126786                    |
| NEO_Agreeableness | 0.136938                    |


<b>Top Features by Feature Importance - Hallucinogens</b>
| Feature           | Impurity-based Importance   |
|:-----------------:|:---------------------------:|
| Country           | 0.295012                    |
| Age               | 0.273682                    |
| Sensation Seeking | 0.206501                    |
| Education         | 0.137931                    |

### Results for Research Question Two - Best Algorithm to Predict Drug Use

#### Initial Model Results - Balanced and Imbalanced Data Comparison

<b>Initial Models for Stimulants</b>
<table>
    <thead>
        <tr>
            <th></th>
            <th colspan=2>Support Vector Machine (SVM)</th>
            <th colspan=2>Logistic Regression</th>
            <th colspan=2>Random Forest</th>
            <th colspan=2>Neural Network </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced </td>
        </tr>
        <tr>
            <td>Mean Accuracy</td>
            <td>0.2068</td>
            <td>0.3050</td>
            <td>0.9938</td>
            <td>0.60212</td>
            <td>0.9934</td>
            <td>0.8820</td>
            <td>0.9934</td>
            <td>0.9125 </td>
        </tr>
    </tbody>
</table>

<b>Initial Models for Depressants</b>
<table>
    <thead>
        <tr>
            <th></th>
            <th colspan=2>Support Vector Machine (SVM)</th>
            <th colspan=2>Logistic Regression</th>
            <th colspan=2>Random Forest</th>
            <th colspan=2>Neural Network </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td> Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced </td>
            <td></td>
        </tr>
        <tr>
            <td>Mean Accuracy</td>
            <td>0.8488</td>
            <td>0.44694</td>
            <td>0.85013</td>
            <td>0.42175</td>
            <td>0.8488</td>
            <td>0.63395</td>
            <td>0.85145</td>
            <td>0.67771</td>
        </tr>
    </tbody>
</table>

<b>Initial Models for Hallucinogens</b>
<table>
    <thead>
        <tr>
            <th></th>
            <th colspan=2>Support Vector Machine (SVM)</th>
            <th colspan=2>Logistic Regression</th>
            <th colspan=2>Random Forest</th>
            <th colspan=2>Neural Network </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td></td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced</td>
            <td>Imbalanced</td>
            <td>Balanced </td>
        </tr>
        <tr>
            <td>Mean Accuracy</td>
            <td>0.4204</td>
            <td>0.24270</td>
            <td>0.69496</td>
            <td>0.66047</td>
            <td>0.70689</td>
            <td>0.67374</td>
            <td>0.67241</td>
            <td>0.684350 </td>
        </tr>
    </tbody>
</table>

#### Final Model Results - Best Models

<b>Final Models for Stimulants</b>

| Metrics          | Support Vector Machine (SVM) | Logistic Regression | Random Forest | Neural Network**  |
|:----------------:|:----------------------------:|:-------------------:|:-------------:|:-----------------:|
| Overall Accuracy | 0.873468                     | 0.755832            | 0.906682      | 0.946224          |
| Precision        | 0.862176                     | 0.742619            | 0.926640      | 0.944476          |
| Recall           | 0.849348                     | 0.676157            | 0.866251      | 0.933719          |
| F1 - Score       | 0.855166                     | 0.688689            | 0.887821      | 0.938777          |
| AUC              | 0.91564                      | 0.8575              | 0.9892        | 0.9801            |
| Timing           | 0.035 s                      | 0.143 s             | 1.505 s       | 2.415 s           |

<p style="background-color:DodgerBlue;"><i>** Best Model</i></p>

<b>Final Models for Depressants</b>

| Metrics          | Support Vector Machine (SVM) | Logistic Regression | Random Forest** | Neural Network  |
|:----------------:|:----------------------------:|:-------------------:|:---------------:|:---------------:|
| Overall Accuracy | 0.717602                     | 0.685148            | 0.844039        | 0.706108        |
| Precision        | 0.683079                     | 0.724233            | 0.851524        | 0.667796        |
| Recall           | 0.633874                     | 0.533807            | 0.790399        | 0.664131        |
| F1 - Score       | 0.640769                     | 0.475795            | 0.809779        | 0.665795        |
| AUC              | 0.7293                       | 0.70741             | 0.9281          | 0.73894         |
| Timing           | 0.0689 s                     | 0.08719 s           | 1.536 s         | 0.6619 s        |

<p style="background-color:DodgerBlue;"><i>** Best Model</i></p>

<b>Final Models for Hallucinogens</b>

| Metrics          | Support Vector Machine (SVM) | Logistic Regression | Random Forest** | Neural Network  |
|:----------------:|:----------------------------:|:-------------------:|:---------------:|:---------------:|
| Overall Accuracy | 0.753274                     | 0.693705            | 0.783270        | 0.756232        |
| Precision        | 0.736559                     | 0.757249            | 0.791162        | 0.735555        |
| Recall           | 0.675539                     | 0.545944            | 0.703105        | 0.685678        |
| F1 - Score       | 0.687642                     | 0.497090            | 0.720170        | 0.697635        |
| AUC              | 0.79457                      | 0.74284             | 0.866084        | 0.8045          |
| Timing           | 0.0695 s                     | 0.5953 s            | 6.0910 s        | 4.1214 s        |

<p style="background-color:DodgerBlue;"><i>** Best Model</i></p>

## Conclusion

This project contributes to the literature and work on predicting drug use outcomes by demonstrating the use of machine learning algorithms in predicting drug use accurately. Being able to more accurately predict drug use can help to inform health interventions and to improve their clinical effectiveness. Understanding the drivers of drug use also provide opportunities for public health officials and policy makers to enact programs and policies that help provide supportive and safe environments that aid individuals in reducing or completely stopping their usage of drugs. This study also provided insight on personality traits and drug use, which has not been a commonly studied association. This work supports the existing body of work and understanding of drug use. The use of neural networks and support vector machines to this dataset were novel approaches. As well, the addition of the OneVsRest classifier to augment the four algorithms to deal with the drug outcomes as multi-class outcomes instead of just binary outcomes was an approach not found previously in the literature. In summary, this study helps to highlight the potential that machine learning algorithms have in assisting in health research and drug use prediction.

Future work should focus on gathering larger amounts of data to address the sample size issues and imbalanced classes found in this study. Additionally, a much larger dataset would help maximize the performance abilities of machine learning algorithms such as neural networks and random forests in producing a more robust and accurate estimate. A larger number of features also need to be retrieved to help provide more predictors to help improve the models and decrease overall model bias. Once the three border outcomes of stimulants, depressants and hallucinogens can be predicted accurately, future studies can then look to predicting very specific drug outcomes, as was found in original dataset.



## Creators & Credits <a name="creators"></a>

* Andrew Leung
    - [https://github.com/as2leung](https://github.com/as2leung)
* Udacity
     - [Website](https://www.udacity.com/)
