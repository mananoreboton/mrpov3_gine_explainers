Context and Motivation

The discovery of antiviral compounds is a central challenge for current and future public health threats. This thesis is framed within the Next Pandemics project, a multidisciplinary initiative focused on the development of computational tools to support drug discovery against present and future pandemics. In this context, machine learning methods can provide systematic evidence for the study of molecular activity, helping prioritize candidate molecules and reducing experimental cost.

This work focuses on compound binding affinity classification for SARS CoV 2 M pro database. Binding affinity describes how strongly a compound interacts with a biological target. In cheminformatics datasets this activity can be represented by a continuous pIC50 value, which summarizes the inhibitory concentration of a compound in a transformed scale. Although pIC50 is continuous, this thesis considers the classification setting, where compounds are assigned to activity classes from their pIC50 values, to distinguishing compounds with different levels of potential relevance for antiviral discovery. 

Compounds are molecules that can naturally be represented as graphs, where atoms are nodes and chemical bonds are edges. Graph Neural Networks (GNN) are therefore well suited for learning from molecular structures and for the subsequent classification of these compounds. However, predictive accuracy alone does not provide insight into why the model makes its predictions. To reveal the factors driving the model's predictions, it is necessary to use explainability methods.

In this context, explainability assesses whether the model relies on chemically meaningful evidence and provides insights that can guide downstream experimental decisions. Numerous graph-based explainers have been proposed in the literature. However, they can yield different explanations for the same molecule and model due to differences in their underlying assumptions and optimization criteria. Consequently, the choice of explainer can influence the interpretation of the model. A comparative study is therefore needed to evaluate how different explainers behave under the same experimental conditions.

However, limitations can arise when evaluating explainers, particularly when ground-truth explanations are unavailable. In this work, we focus on this common scenario. In such cases, there is no definitive annotation identifying which atoms or bonds are responsible for the observed activity. As a result, evaluation cannot rely on direct comparison with known substructures and instead requires quantitative metrics that capture desirable properties of explanations.

The motivation of this thesis is to compare metrics for evaluating GNN explainers in the absence of ground-truth explanations. The predictive model is based on GINE, a graph neural network architecture that incorporates edge information during message passing. This choice is treated as fixed, as the focus of this work is not on comparing predictive architectures but on evaluating explanations produced by a trained graph model. The study is delimited to the SARS-CoV-2 Mpro dataset and a three-class pIC50 classification task. Within this scope, the thesis aims to clarify how different evaluation metrics characterize explainer behavior and how they can support a more rigorous interpretation of GNN explanations in cheminformatics.





