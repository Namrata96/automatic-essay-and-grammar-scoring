# Automatic Essay and Grammar scoring
This work was an adaptation of [1] to predict grammar scores using a bi-directional LSTM (biLSTM) network instead of overall
essay scores which the paper originally predicted. The paper also involved training score-specific word embeddings
which would take into account the usage information of a word (spelling errors are informative and prepositions are
not). We adapted the same to learn a representation that treats grammatical errors as informative. In this project, I
developed an end-to-end pipeline that would take an essay, generate an essay embedding with the augmented C&W [2]
word embeddings and biLSTM, and then the final layer would predict the grammar score. 

### References
1. D. Alikaniotis, H. Yannakoudakis, and M. Rei, “Automatic text scoring using neural networks,” in Proceedings of the 54th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), 2016, pp. 715–725. Available <a href="https://www.aclweb.org/anthology/P16-1068.pdf">here</a>.
2. R. Collobert and J. Weston, “A unified architecture for natural language processing: Deep neural networks with multitask learning,” in
Proceedings of the 25th International Conference on Machine Learning, ser. ICML ’08. New York, NY, USA: ACM, 2008, pp. 160–167.
[Online]. Available <a href="http://doi.acm.org/10.1145/1390156.1390177">here</a>.
