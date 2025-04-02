#! /usr/bin/env python


import keyring
from openai import OpenAI


SERVICE = 'OpenAI'
NAME = 'HomeworkAPIKey'


if __name__ == '__main__':
    apikey = keyring.get_password(SERVICE, NAME)
    client = OpenAI(api_key=apikey)

    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        store=True,
        messages=[
            {'role': 'user', 'content': 'write a haiku about ai'}
        ]
    )

    print(completion.choices[0].message)
