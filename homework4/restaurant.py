#! /usr/bin/env python


import sys

import streamlit as st

import langchain_helper


if __name__ == '__main__':
    st.title('Restaurant Name Generator')
    if cuisine := st.sidebar.selectbox(
        'Pick a Cuisine',
        [
            'American',
            'Arabic',
            'Chinese',
            'Indian',
            'Italian',
            'Japanese',
            'Mexican',
            'Polish',
            'Thai',
        ],
    ):
        response = langchain_helper.get_restaurant_name_and_items(cuisine)
        st.header(response['restaurant_name'])
        menu_items = response['menu_items']
        st.write('**Menu Items:**')

        # Doesn't work correctly - results in nested lists:
        # for item in menu_items:
        #     st.write(f'- {item}')
        # Note:  Can't "comment" out with triple quotes - streamlit interprets this
        #        as HTML text.

        # Build a single Markdown string with numbers
        menu_md = '\n'.join(f'{i}. {item}' for i, item in enumerate(menu_items, start=1))
        # Alternatively could use f"- {item}" ... for bullets

        # Fix quirk in streamlit where it sometimes prepends an addition '1. ' to
        # menu_md resulting in a nested list and undesired output:
        if menu_md.startswith('1. 1.'):
            menu_md = menu_md.replace('1. 1.', '1. ')

        # Sometimes get this:
        # '*/\n            Console.WriteLine("Tacos', 'Enchiladas', 'Quesadillas', 'Tamales',
        # 'Chilaquiles"); \n        }\n    }\n}'
        # Could do regexp to extract out comma separated values...

        # Debugging:
        print(f'\nFor {cuisine=}:\n{menu_items=}\n{menu_md=}\n', file=sys.stderr)

        # Render as one list
        st.markdown(menu_md)
