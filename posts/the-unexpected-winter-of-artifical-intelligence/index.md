<!--
.. title: The unexpected winter of Artifical Intelligence
.. slug: the-unexpected-winter-of-artifical-intelligence
.. date: 2023-03-05 07:13:36 UTC+01:00
.. tags: deep-learning, opinion, ai
.. category: 
.. link: 
.. description: 
.. type: text#
.. has_math: true
.. medium: yes
-->

*Nowadays, everyone is exicted the latest trends in AI applications, like ChatGPT, self-driving cars, Image synthesis, etc. This overhype is not new, it happened before in the first AI winter in the 1980s. Some warn againt it, because it may cause dispointment and even a new AI winter. But here I will talk about the bottelneck of AI research that I come across in my work. It may be not be called a winter, but it's difinitely will cause a slow down in the field.*


# What is wrong? Too many models

When a newcomer wants to review tha state of the art in a machine learning field, like objects detection, video understanding, or image synthesis, he will find a lot of models. One approch is to pick just the most notable work in the current year, but this also means that he will miss a lot of other work that may be more suitable for his problem. Another approch is to review all the models, but this is a lot of work, and it's not clear how to compare them.

In the voice of this poor overwhilmed researcher, I would say: *"Please, enough with the models, and more coherent thoeories that could add up!"*.

# Research Question

When the research question for the majorty of these papers is viewed, one can see that they have a 'template' reseach question, like:

* How to improve the performance of the model?
* How to improve the speed of the model?
* How to improve the accuracy of the model?

but questions like that should be stated more like:

* How to solve the performance problems of the last state of the art model?

Because it's not enough to show a better numeric values as an 'answer'. It should be more detailed to when and why the improvment happens. This is diffently much helpful than just saying that the model outperforms the previous state of the art.


# Research Importance

It's known that the deep learning field is expermintal. But if the whole contribution is training and evaulating a model, then this is like saying "I proved it in condition X. Every other condition is not tested yet". This actually helps no one. The contribution should be more than just a technical improvment. It should be a new idea that can be used in other fields, or a new way to solve a problem. 

# What to do?

## Less is more

I think this can start from every researcher. Where making a model should be based on robust math foundations. I'm not saying that should enhance the quality over quantity of the field, but it will be easier to review, because a newcomer could easily fit every new work into the whole picture, and know exactly what is needed for what in every situation.

## Math-only models

In fact, the new models can be proven without training. It's a brave claim, I know. But I can give some examples, like Kalman filer paper [1], where the kalman filter model is proven to be optimal in the linear case only mathimatically.  Another example is "Visual SLAM, why filter?" [2] where partly the proof is mathmatical for the superiority of bundle adjustment method over kalman filtering method. 

If you have a good knowledge in the fields of robotics, you will know that these papers are really famous. So one might say that, not everyone can do that. But why not try at least?. Or just make it clear that the model is not proven, and it's just a good guess while also making effort to explore the theory behind it. 


## Lastly, GPUs for all

I think that the main reason for the overhype is the availability of GPUs. It's not a secret that GPUs are the main reason for the recent AI boom. But it's not a secret that strong GPUs are not available for everyone. 

Therefore, the ones who has the best hardware, will generate better models. This deviate the reasearch from its goals.
One soultion could be to provide GPU access for all researchers, like Google is doing [3]. But this could be done in more broad way that grauntate the elimination of hardware hinderance amonge the researchers.

If one day, all the reseach community used one gloabl system, this will also ease the reproduction of the work.


# Note:

This article is my personal opnion based on my limited experince in the field. As I'm still learning, I'm open to any feedback and discussion.

# References

[1] https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction?redirectedFrom=fulltext

[2] https://www.sciencedirect.com/science/article/abs/pii/S0262885612000248

[3] https://edu.google.com/programs/credits/research/?modal_active=none