# Test cases that demonstrate bug in GPT parsing non-ascii characters.

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
from neuron_explainer.api_client import ApiClient

api_client = ApiClient(model_name="gpt-3.5-turbo-1106", max_concurrent=1)

"""
this example gets GPT to respond with fewer tokens than we gave it - it omits the ellipses token/activation.
to get GPT to truncate its response resulting in invalid json, replace to_send.activations with:
    "activations": [
        {"token": "hello", "activation": None},
        {"token": "hello", "activation": None},
        {"token": "hello", "activation": None},
        {"token": "hello", "activation": None},
        {"token": "hello", "activation": None},
        {"token": "hello", "activation": None},
        {"token": " …", "activation": None},
        {"token": " \u2022", "activation": None},
        {"token": " £", "activation": None},
    ]
"""
to_send = {
    "neuron": 3,
    "explanation": "'protect', 'know', 'with' and 'save'",
    "activations": [
        {"token": " on", "activation": None},
        {"token": " some", "activation": None},
        {"token": " days", "activation": None},
        {"token": " we", "activation": None},
        {"token": " post", "activation": None},
        {"token": " an", "activation": None},
        {"token": " afternoon", "activation": None},
        {"token": " story", "activation": None},
        {"token": " at", "activation": None},
        {"token": " around", "activation": None},
        {"token": " 2", "activation": None},
        {"token": " PM", "activation": None},
        {"token": ".", "activation": None},
        {"token": " After", "activation": None},
        {"token": " every", "activation": None},
        {"token": " new", "activation": None},
        {"token": " story", "activation": None},
        {"token": " we", "activation": None},
        {"token": " send", "activation": None},
        {"token": " out", "activation": None},
        {"token": " an", "activation": None},
        {"token": " alert", "activation": None},
        {"token": " to", "activation": None},
        {"token": " our", "activation": None},
        {"token": " e", "activation": None},
        {"token": "-", "activation": None},
        {"token": "mail", "activation": None},
        {"token": " list", "activation": None},
        {"token": " and", "activation": None},
        {"token": " our", "activation": None},
        {"token": " FB", "activation": None},
        {"token": " page", "activation": None},
        {"token": ".", "activation": None},
        {"token": "\n", "activation": None},
        {"token": "\n", "activation": None},
        {"token": "Learn", "activation": None},
        {"token": " about", "activation": None},
        {"token": " Scientology", "activation": None},
        {"token": " with", "activation": None},
        {"token": " our", "activation": None},
        {"token": " numerous", "activation": None},
        {"token": " series", "activation": None},
        {"token": " with", "activation": None},
        {"token": " experts", "activation": None},
        {"token": "…", "activation": None},
        {"token": "\n", "activation": None},
        {"token": "\n", "activation": None},
        {"token": "BL", "activation": None},
        {"token": "OG", "activation": None},
        {"token": "G", "activation": None},
        {"token": "ING", "activation": None},
        {"token": " DI", "activation": None},
        {"token": "AN", "activation": None},
        {"token": "ET", "activation": None},
        {"token": "ICS", "activation": None},
        {"token": ":", "activation": None},
        {"token": " We", "activation": None},
        {"token": " read", "activation": None},
        {"token": " Scientology", "activation": None},
        {"token": "��", "activation": None},
        {"token": "s", "activation": None},
        {"token": " founding", "activation": None},
        {"token": " text", "activation": None},
    ],
}

prompt = [
    {
        "role": "system",
        "content": "We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at an explanation of what the neuron does, and try to predict its activations on a particular token.\n\nFor each sequence, you will see the tokens in the sequence where the activations are left blank. You will print, in valid json, the exact same tokens verbatim, but with the activation values filled in according to the explanation.\nFill out the activation values from 0 to 10. Most activations will be 0.\n",
    },
    {
        "role": "user",
        "content": '{"neuron": 1, "explanation": "language related to something being groundbreaking", "activations": [{"token": "The", "activation": None}, {"token": " editors", "activation": None}, {"token": " of", "activation": None}, {"token": " Bi", "activation": None}, {"token": "opol", "activation": None}, {"token": "ym", "activation": None}, {"token": "ers", "activation": None}, {"token": " are", "activation": None}, {"token": " delighted", "activation": None}, {"token": " to", "activation": None}, {"token": " present", "activation": None}, {"token": " the", "activation": None}, {"token": " ", "activation": None}, {"token": "201", "activation": None}, {"token": "8", "activation": None}, {"token": " Murray", "activation": None}, {"token": " Goodman", "activation": None}, {"token": " Memorial", "activation": None}, {"token": " Prize", "activation": None}, {"token": " to", "activation": None}, {"token": " Professor", "activation": None}, {"token": " David", "activation": None}, {"token": " N", "activation": None}, {"token": ".", "activation": None}, {"token": " Ber", "activation": None}, {"token": "atan", "activation": None}, {"token": " in", "activation": None}, {"token": " recognition", "activation": None}, {"token": " of", "activation": None}, {"token": " his", "activation": None}, {"token": " seminal", "activation": None}, {"token": " contributions", "activation": None}, {"token": " to", "activation": None}, {"token": " bi", "activation": None}, {"token": "oph", "activation": None}, {"token": "ysics", "activation": None}, {"token": " and", "activation": None}, {"token": " their", "activation": None}, {"token": " impact", "activation": None}, {"token": " on", "activation": None}, {"token": " our", "activation": None}, {"token": " understanding", "activation": None}, {"token": " of", "activation": None}, {"token": " charge", "activation": None}, {"token": " transport", "activation": None}, {"token": " in", "activation": None}, {"token": " biom", "activation": None}, {"token": "olecules", "activation": None}, {"token": ".\\n\\n", "activation": None}, {"token": "In", "activation": None}, {"token": "aug", "activation": None}, {"token": "ur", "activation": None}, {"token": "ated", "activation": None}, {"token": " in", "activation": None}, {"token": " ", "activation": None}, {"token": "200", "activation": None}, {"token": "7", "activation": None}, {"token": " in", "activation": None}, {"token": " honor", "activation": None}, {"token": " of", "activation": None}, {"token": " the", "activation": None}, {"token": " Bi", "activation": None}, {"token": "opol", "activation": None}, {"token": "ym", "activation": None}, {"token": "ers", "activation": None}, {"token": " Found", "activation": None}, {"token": "ing", "activation": None}, {"token": " Editor", "activation": None}, {"token": ",", "activation": None}, {"token": " the", "activation": None}, {"token": " prize", "activation": None}, {"token": " is", "activation": None}, {"token": " awarded", "activation": None}, {"token": " for", "activation": None}, {"token": " outstanding", "activation": None}, {"token": " accomplishments", "activation": None}]}',
    },
    {
        "role": "assistant",
        "content": '{"neuron": 1, "explanation": "language related to something being groundbreaking", "activations": [{"token": "The", "activation": 0}, {"token": " editors", "activation": 0}, {"token": " of", "activation": 0}, {"token": " Bi", "activation": 0}, {"token": "opol", "activation": 0}, {"token": "ym", "activation": 0}, {"token": "ers", "activation": 0}, {"token": " are", "activation": 0}, {"token": " delighted", "activation": 0}, {"token": " to", "activation": 0}, {"token": " present", "activation": 0}, {"token": " the", "activation": 0}, {"token": " ", "activation": 0}, {"token": "201", "activation": 0}, {"token": "8", "activation": 0}, {"token": " Murray", "activation": 0}, {"token": " Goodman", "activation": 0}, {"token": " Memorial", "activation": 0}, {"token": " Prize", "activation": 0}, {"token": " to", "activation": 0}, {"token": " Professor", "activation": 0}, {"token": " David", "activation": 0}, {"token": " N", "activation": 0}, {"token": ".", "activation": 0}, {"token": " Ber", "activation": 0}, {"token": "atan", "activation": 0}, {"token": " in", "activation": 0}, {"token": " recognition", "activation": 0}, {"token": " of", "activation": 0}, {"token": " his", "activation": 0}, {"token": " seminal", "activation": 10}, {"token": " contributions", "activation": 0}, {"token": " to", "activation": 0}, {"token": " bi", "activation": 0}, {"token": "oph", "activation": 0}, {"token": "ysics", "activation": 0}, {"token": " and", "activation": 0}, {"token": " their", "activation": 0}, {"token": " impact", "activation": 0}, {"token": " on", "activation": 0}, {"token": " our", "activation": 0}, {"token": " understanding", "activation": 0}, {"token": " of", "activation": 0}, {"token": " charge", "activation": 0}, {"token": " transport", "activation": 0}, {"token": " in", "activation": 0}, {"token": " biom", "activation": 0}, {"token": "olecules", "activation": 0}, {"token": ".\\n\\n", "activation": 0}, {"token": "In", "activation": 0}, {"token": "aug", "activation": 0}, {"token": "ur", "activation": 0}, {"token": "ated", "activation": 0}, {"token": " in", "activation": 0}, {"token": " ", "activation": 0}, {"token": "200", "activation": 0}, {"token": "7", "activation": 0}, {"token": " in", "activation": 0}, {"token": " honor", "activation": 0}, {"token": " of", "activation": 0}, {"token": " the", "activation": 0}, {"token": " Bi", "activation": 0}, {"token": "opol", "activation": 0}, {"token": "ym", "activation": 0}, {"token": "ers", "activation": 0}, {"token": " Found", "activation": 0}, {"token": "ing", "activation": 1}, {"token": " Editor", "activation": 0}, {"token": ",", "activation": 0}, {"token": " the", "activation": 0}, {"token": " prize", "activation": 0}, {"token": " is", "activation": 0}, {"token": " awarded", "activation": 0}, {"token": " for", "activation": 0}, {"token": " outstanding", "activation": 0}, {"token": " accomplishments", "activation": 0}]}',
    },
    {
        "role": "user",
        "content": '{"neuron": 2, "explanation": "the word \\u201cvariant\\u201d and other words with the same \\u201dvari\\u201d root", "activations": [{"token": "{\\"", "activation": None}, {"token": "widget", "activation": None}, {"token": "Class", "activation": None}, {"token": "\\":\\"", "activation": None}, {"token": "Variant", "activation": None}, {"token": "Matrix", "activation": None}, {"token": "Widget", "activation": None}, {"token": "\\",\\"", "activation": None}, {"token": "back", "activation": None}, {"token": "order", "activation": None}, {"token": "Message", "activation": None}, {"token": "\\":\\"", "activation": None}, {"token": "Back", "activation": None}, {"token": "ordered", "activation": None}, {"token": "\\",\\"", "activation": None}, {"token": "back", "activation": None}, {"token": "order", "activation": None}, {"token": "Message", "activation": None}, {"token": "Single", "activation": None}, {"token": "Variant", "activation": None}, {"token": "\\":\\"", "activation": None}, {"token": "This", "activation": None}, {"token": " item", "activation": None}, {"token": " is", "activation": None}, {"token": " back", "activation": None}, {"token": "ordered", "activation": None}, {"token": ".\\",\\"", "activation": None}, {"token": "ordered", "activation": None}, {"token": "Selection", "activation": None}, {"token": "\\":", "activation": None}, {"token": "true", "activation": None}, {"token": ",\\"", "activation": None}, {"token": "product", "activation": None}, {"token": "Variant", "activation": None}, {"token": "Id", "activation": None}, {"token": "\\":", "activation": None}, {"token": "0", "activation": None}, {"token": ",\\"", "activation": None}, {"token": "variant", "activation": None}, {"token": "Id", "activation": None}, {"token": "Field", "activation": None}, {"token": "\\":\\"", "activation": None}, {"token": "product", "activation": None}, {"token": "196", "activation": None}, {"token": "39", "activation": None}, {"token": "_V", "activation": None}, {"token": "ariant", "activation": None}, {"token": "Id", "activation": None}, {"token": "\\",\\"", "activation": None}, {"token": "back", "activation": None}, {"token": "order", "activation": None}, {"token": "To", "activation": None}, {"token": "Message", "activation": None}, {"token": "Single", "activation": None}, {"token": "Variant", "activation": None}, {"token": "\\":\\"", "activation": None}, {"token": "This", "activation": None}, {"token": " item", "activation": None}, {"token": " is", "activation": None}, {"token": " back", "activation": None}, {"token": "ordered", "activation": None}, {"token": " and", "activation": None}, {"token": " is", "activation": None}, {"token": " expected", "activation": None}, {"token": " by", "activation": None}, {"token": " {", "activation": None}, {"token": "0", "activation": None}, {"token": "}.", "activation": None}, {"token": "\\",\\"", "activation": None}, {"token": "low", "activation": None}, {"token": "Price", "activation": None}, {"token": "\\":", "activation": None}, {"token": "999", "activation": None}, {"token": "9", "activation": None}, {"token": ".", "activation": None}, {"token": "0", "activation": None}, {"token": ",\\"", "activation": None}, {"token": "attribute", "activation": None}, {"token": "Indexes", "activation": None}, {"token": "\\":[", "activation": None}, {"token": "],\\"", "activation": None}, {"token": "productId", "activation": None}, {"token": "\\":", "activation": None}, {"token": "196", "activation": None}, {"token": "39", "activation": None}, {"token": ",\\"", "activation": None}, {"token": "price", "activation": None}, {"token": "V", "activation": None}, {"token": "ariance", "activation": None}, {"token": "\\":", "activation": None}, {"token": "true", "activation": None}, {"token": ",\\"", "activation": None}]}',
    },
    {
        "role": "assistant",
        "content": '{"neuron": 2, "explanation": "the word \\u201cvariant\\u201d and other words with the same \\u201dvari\\u201d root", "activations": [{"token": "{\\"", "activation": 0}, {"token": "widget", "activation": 0}, {"token": "Class", "activation": 0}, {"token": "\\":\\"", "activation": 0}, {"token": "Variant", "activation": 6}, {"token": "Matrix", "activation": 0}, {"token": "Widget", "activation": 0}, {"token": "\\",\\"", "activation": 0}, {"token": "back", "activation": 0}, {"token": "order", "activation": 0}, {"token": "Message", "activation": 0}, {"token": "\\":\\"", "activation": 0}, {"token": "Back", "activation": 0}, {"token": "ordered", "activation": 0}, {"token": "\\",\\"", "activation": 0}, {"token": "back", "activation": 0}, {"token": "order", "activation": 0}, {"token": "Message", "activation": 0}, {"token": "Single", "activation": 0}, {"token": "Variant", "activation": 0}, {"token": "\\":\\"", "activation": 0}, {"token": "This", "activation": 0}, {"token": " item", "activation": 0}, {"token": " is", "activation": 0}, {"token": " back", "activation": 0}, {"token": "ordered", "activation": 0}, {"token": ".\\",\\"", "activation": 0}, {"token": "ordered", "activation": 0}, {"token": "Selection", "activation": 0}, {"token": "\\":", "activation": 0}, {"token": "true", "activation": 0}, {"token": ",\\"", "activation": 0}, {"token": "product", "activation": 0}, {"token": "Variant", "activation": 0}, {"token": "Id", "activation": 0}, {"token": "\\":", "activation": 0}, {"token": "0", "activation": 0}, {"token": ",\\"", "activation": 0}, {"token": "variant", "activation": 0}, {"token": "Id", "activation": 0}, {"token": "Field", "activation": 0}, {"token": "\\":\\"", "activation": 0}, {"token": "product", "activation": 0}, {"token": "196", "activation": 0}, {"token": "39", "activation": 0}, {"token": "_V", "activation": 0}, {"token": "ariant", "activation": 0}, {"token": "Id", "activation": 0}, {"token": "\\",\\"", "activation": 0}, {"token": "back", "activation": 0}, {"token": "order", "activation": 0}, {"token": "To", "activation": 0}, {"token": "Message", "activation": 0}, {"token": "Single", "activation": 0}, {"token": "Variant", "activation": 0}, {"token": "\\":\\"", "activation": 0}, {"token": "This", "activation": 0}, {"token": " item", "activation": 0}, {"token": " is", "activation": 0}, {"token": " back", "activation": 0}, {"token": "ordered", "activation": 0}, {"token": " and", "activation": 0}, {"token": " is", "activation": 0}, {"token": " expected", "activation": 0}, {"token": " by", "activation": 0}, {"token": " {", "activation": 0}, {"token": "0", "activation": 0}, {"token": "}.", "activation": 0}, {"token": "\\",\\"", "activation": 0}, {"token": "low", "activation": 0}, {"token": "Price", "activation": 0}, {"token": "\\":", "activation": 0}, {"token": "999", "activation": 0}, {"token": "9", "activation": 0}, {"token": ".", "activation": 0}, {"token": "0", "activation": 0}, {"token": ",\\"", "activation": 0}, {"token": "attribute", "activation": 0}, {"token": "Indexes", "activation": 0}, {"token": "\\":[", "activation": 0}, {"token": "],\\"", "activation": 0}, {"token": "productId", "activation": 0}, {"token": "\\":", "activation": 0}, {"token": "196", "activation": 0}, {"token": "39", "activation": 0}, {"token": ",\\"", "activation": 0}, {"token": "price", "activation": 0}, {"token": "V", "activation": 0}, {"token": "ariance", "activation": 1}, {"token": "\\":", "activation": 0}, {"token": "true", "activation": 0}, {"token": ",\\"", "activation": 0}]}',
    },
    {
        "role": "user",
        "content": json.dumps(to_send),
    },
]


print("activations length sent to GPT: " + str(len(to_send["activations"])))
print("activation tokens sent: ")
print([activation["token"] for activation in to_send["activations"]])


async def run():
    response = await api_client.make_request(
        messages=prompt, max_tokens=2000, temperature=0, json_mode=True
    )
    choice = response["choices"][0]
    completion = choice["message"]["content"]
    received_json = json.loads(completion)
    print(
        "activations length received from GPT: "
        + str(len(received_json["activations"]))
    )
    print([activation["token"] for activation in received_json["activations"]])


asyncio.run(run())
