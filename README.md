# GPT2 

This repo is inspired by **Andrej Karpathy's** YouTube video: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=12016s). I've tried to explain to concepts related to GPT2 which I learnt through the video.

Here we are not developing the chat functionality as you seen in ChatGPT, which is done after pretraining of the models. But we can fine-tune the modeL using SFT and system-user chat dataset. Here we are developing next token predicition using GPT2 model from scrartch. Heres the example:

```
Hello, I'm a language modeler. I'm fluent in several languages. I know a lot about each one, but I don't usually write
Hello, I'm a language modeler, not a compiler and never got the time to learn this idea. I've also been working on my own
Hello, I'm a language model, but there's so little interaction that those things get out of hand at different moments, so that has to be
Hello, I'm a language model, one of the key tenets of the open source, open source computing ecosystem. I'm the founder, co-
Hello, I'm a language model. I take for granted the limitations of the English-speaking world and the vast resources of knowledge it offers. I
 ```

# Decoder only Transformer
In **Attention is all you need** paper the researcher used both encoder and decoder but for machine translation. Here we needed encoder to encode the semantics and contextual info present in English language. Decoder was given the ability to use the encoder knowlege as well as the mapping between the english and translated language to translate the language. This mapping is nothing but **Cross Attention**

## Attention
In transformer paradigm there are 3 major attention types:

- Self Attention
- Cross Attention
- Causal Attention

### Self-Attention
As the word suggests, in self-attention the words attend themesleves, like one to many mapping. Consider input set:

$$
\{x_i\}_{i=1}^{i=t} = \{x_1,\ldots,x_t\}
$$

$$
X \in \mathbb{R}^{n \times t}
$$

In the context of self-attention, the hidden state is formed by combining the input tokens $x_1, x_2\ldots,x_t$ with the corresponding attention weights $\alpha_1,\alpha_2\ldots,\alpha_t$
$$h = \alpha_1 x_1 + \alpha_2 x_2 + \cdots + \alpha_t x_t$$
In self-attention, the model computes the attention weights $\alpha_i$ based on the relationship between tokens in the sequence. This allows the model to focus on different parts of the input sequence when forming the hidden representation.

$$
a = [\text{soft}](\text{arg})\max_\beta(X^\top x)
$$

But, how this $\alpha_i$ is calculated?
- For each input vector $x_i$ we calculate how similar it is to all the other input vectors. We do this by taking dot product of $x$ with each vector in the set. This gives the similarity score.
- After this, we apply softmax function to these similarity scores. This turns the score into probabilities that sum to 1. These probabilities become our attention weights. 
- Doing this for all vectors creates a set of attention weights. These sets of attention weights form the rows of our attention matrix $A$.

#### Query Keys and Value
In previous section we saw how to calculate attention score. But we there are trainabale parameters for our model to learn for these similarity scores. Lets introduce them:

$$
q = W_q x\\k=W_kx\\v = W_v x
$$

In the given above equations, we can think of projecting these input vectors into higher dimesions for better representations. Each dimensions can be interpreted as some properties like: semantics, contextual features, syntatic features, etc.

Lets understand this by an example taken from [**Deep Learning notes from NYU**](https://atcold.github.io/NYU-DLSP20/en/week12/12-3/)

> For example, say we wanted to find a recipe to make lasagne. We have a recipe book and search for “lasagne” - this is the query. This query is checked against all possible keys in your dataset - in this case, this could be the titles of all the recipes in the book. We check how aligned the query is with each title to find the maximum matching score between the query and all the respective keys. If our output is the argmax function - we retrieve the single recipe with the highest score. Otherwise, if we use a soft argmax function, we would get a probability distribution and can retrieve in order from the most similar content to less and less relevant recipes matching the query.



We compare the query ***lasagne*** with all keys ***recipe titles*** to get attention scores. These scores determine how much attention to pay to each value ***recipe content***. Using softmax, we get a weighted combination of multiple recipes with weights based on how well each title matched ***lasagne***

$$
a = \text{softmax}\left(\frac{K^\top q}{\sqrt{d}}\right) \in \mathbb{R}^t
$$

The attention weights $a_1, a_2, \ldots$ represent how relevant each token is to the current query.
The attention weights $a$ are then used to compute a weighted sum of the Value (V) vectors. This step aggregates the information from the input sequence based on the relevance determined by the attention weights.
Now this output is very important for us as this contains which tokens are important given the context of input tokens. This projects the Value vectors into new representation. This new representation emphasizes the tokens that are more relvant to the query and de-emphasizes those that are less relevant. Essentially, it *projects* the value vectors into a space where the importance of each token is adjusted based on the attention scores. The resulting vector is a context-aware representation that captures the relationships and dependencies between tokens.

$$
H = VA \in \mathbb{R}^{d \times t}
$$

## Cross Attention
Now, that we understood what is self-attention, lets look at cross-attention. This mechanism is used in Machine Translation, Multimodal tasks etc. This is also known as encoder- decoder attention. Because information flows between encoder and decoder. Unlike in self-attention where we use single input sequence, in cross-attention we mix two different input sequences. The length of input sequence can differ but their dimensions must match. 