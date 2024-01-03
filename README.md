# Neuronpedia Scorer

## Purpose

When players submit an explanation on [Neuronpedia](https://neuronpedia.org), we score it to evaluate how good the explanation is, relevant to the activations. The score is produced is from -1 to 1. This is the first step to fully open-sourcing Neuronpedia. 🎉

**Goals / Desired Outcomes**

- Improve the existing scorer ("see contributing" for ideas) - or fix bugs/issues with it.
- You build a new scorer that is better than one we're using now, and we add it to Neuronpedia and throw a huge party to celebrate.
- Other improvements (enhancements test cases, better docs, etc)

## How It Works

This is a Flask server with only one endpoint that is called by the Neuronpedia server: `POST /score`

The code is mostly plugging in OpenAI's [Automated Interpretability](https://github.com/hijohnnylin/automated-interpretability) Neuron Explainer [demo](https://github.com/openai/automated-interpretability/blob/main/neuron-explainer/demos/generate_and_score_explanation.ipynb). The way _that_ works is that for each explanation, it asks GPT to guess ("simulate") what the activations will be for each of the 20 texts. Then it does a correlation between what the actual activations are vs the simulated ones. The correlation is the score.

It's not perfect, but it works well a lot of the time. For details, including the exact prompts used, [read their paper](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) or [code](https://github.com/openai/automated-interpretability).

## Warning About Cost

Based on a standard 20 activation texts, each text with 64 tokens, and using the default `gpt-4` simulator, it costs a whopping **30 cents** (USD) to do one score for an explanation. So, be careful and make sure to set both a hard and soft limit on your OpenAI account. You could also start with sending the scorer fewer activation texts.

If you end up doing serious batches of work, it may be worth it to apply for [Research Credits](https://openai.com/form/researcher-access-program) from OpenAI.

## Usage

Clone the repository:

```
git clone https://github.com/hijohnnylin/neuronpedia-scorer
```

In the new directory, create a `.env` file that contains:

```
OPENAI_API_KEY="[insert your OpenAI key here]"
SERVER_KEY="[empty string if you aren't hosting it publicly]"
```

Then, run the following to install requirements and start the local Flask server at port 5000:

```
pip install -r requirements.txt

python server.py

> loading
>  * Serving Flask app 'server'
> * Debug mode: off
> WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
> * Running on http://127.0.0.1:5000
> Press CTRL+C to quit
```

Finally, to test that it's working, you can run this curl command, which asks the scorer to score two activation texts: "the quick brown fox" and "spotted leopard sprints at" using the explanation "fast animals".

**WARNING: THIS WILL COST A FEW CENTS**

```
curl -XPOST -H "Content-type: application/json" -d '{ "explanation": "fast animals", "secret": "", "activations": [ { "tokens": ["the ", " quick", " brown", " fox"], "values": [0, 2, 0, 2.5] }, { "tokens": ["spotted ", " leopard", " sprints", " at"], "values": [0.5, 2.5, 1.5, 0] } ] }' 'localhost:5000/score'
```

You should get a response like this, which shows a score of 0.8896 (max is 1), indicating this was a pretty good explanation:

```
{
  "score": 0.8896365998071626,
  "simulations": [array of ScoredSequenceSimulation, see below for spec]
}
```

## Endpoint Spec

`POST /score`

**Request**

```
{
    explanation: "fast animals",
    secret: SERVER_KEY,
    activations: [
        {
            tokens: ["the ", " quick", " brown", " fox"],
            values: [0, 2, 0, 2.5]
        },
        {
            tokens: ["spotted ", " leopard", " sprints", " at"],
            values: [0.5, 2.5, 1.5, 0]
        }
    ]
}
```

**Response**

```
{
    score: the score, a value from from -1 to 1,
    simulations: an array of ScoredSequenceSimulation, which is basically just the simulated activation values that GPT returns
}
```

Here is the [format of ScoredSequenceSimulation](https://github.com/openai/automated-interpretability/blob/8be455788f43a603381e3c1b38a697ad4797a90f/neuron-explainer/neuron_explainer/explanations/explanations.py#L71-L93).

## Contributing and Ideas for Improvement

I'd appreciate any contributions you'd like to make, whether that's bug reports, adding new features, etc.

In general, this code is fairly specific in that it's basically a thin wrapper around OpenAI's Neuron Explainer. The most impactful thing is to make the scorer itself better or to come up with a better scorer: for a set of activation texts, a higher score should seem like a better explanation, and a lower score should seem like a worse explanation. Currently, this is not always the case.

### Examples of Scoring Improvement Needed

Here are a few examples of where the scorer could be improved, mostly around incorporating the context. In these examples, the highest scored explanations are good, but they should probably have a lower score than other explanations that explain the full context around the non-highest-activating tokens.

- [GPT2-SMALL@6:2294](https://www.neuronpedia.org/gpt2-small/6/2294) - top score of 33 is `the words 'chapter' and 'on'`. however, another explanation `words and phrases describing the structure of written media, especially chapters and themes` has a lower score yet explains the full context better

- [GPT2-SMALL@6:281](https://www.neuronpedia.org/gpt2-small/6/1072) - top score is GPT4's `female pronouns and related phrases`, because the token "she" is activated, but it's actually only activated as part of the word "shelled", which has nothing to do with female pronouns - the human explanation by user `turnippls` is far better

- [GPT2-SMALL@6:2381](https://www.neuronpedia.org/gpt2-small/6/2381) - top score is `the word 'time'` but the activations are about spending a portion of time and the other explanation should probably score higher

### Ideas for Improvement

There are at least a few things that can be tried, tested, and benchmarked to improve scoring accuracy - the following are highly promising but I haven't had time to try.

1. **Select Different Activation Text Samples** - We currently select the top 20 activating texts, each with 64 tokens, to give to the simulator - aka the "top k" approach. We should try variations of this - top 10 plus 10 random ones, or top 20 but prefer activations with different highest activating tokens, etc.
2. **Modify the GPT Simulation Prompts** - The GPT simulation prompts are pretty interesting. But like every piece of software, there can probably be improvements. Maybe you can specifically get the prompt to weigh context as more important than just the top activating token. You can see the prompts used in [their paper under 'Step 2: Simulation'](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html#sec-algorithm-simulate). As a bonus, this might result in cost savings if the prompts you come up with are shorter, or if they can be used with GPT-3.5-Turbo (without logprobs), instead of the expensive "text-davinci-003" model used now.
3. **Add a "Post-Scoring Adjustment"** - Maybe after the score/correlation is calculated with the Neuron Explainer scorer, you can then run another prompt or algorithm that adjusts the score based on something you want to prioritize more, like context.
4. **Create an Entirely New Scorer** - Make up your own scorer! We'll plug it into Neuronpedia and have users play with it. You will of course be credited. Some ideas - wanna train your own model? Do a regex scorer? Regex + GPT4 magic sauce? Got some clever dictionary approach?

## Credits

[OpenAI Automated Interpretability](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)

## License

MIT
