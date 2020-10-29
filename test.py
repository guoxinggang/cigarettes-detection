import streamlit as st
from PIL import Image
import time

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')


def main():
    # st.write('# Experimental streamlit layout style!')
    # st.sidebar.markdown('# function')
    # col1, col2 = st.beta_columns(2)
    # original = Image.open('picture/cat.jpg')
    # col1.header("Original")
    # col1.image(original, use_column_width=True)
    # grayscale = original.convert('LA')
    # col2.header("Grayscale")
    # col2.image(grayscale, use_column_width=True)

    # st.title("Let's create a table!")
    # for i in range(1, 10):
    #     cols = st.beta_columns(4)
    #     cols[0].write(f'{i}')
    #     cols[1].write(f'{i * i}')
    #     cols[2].write(f'{i * i * i}')
    #     cols[3].write('x' * i)

    # # Use the full page instead of a narrow central column
    # st.beta_set_page_config(layout="wide")
    # st.write('# Experimental streamlit layout style!')
    # # Space out the maps so the first one is 2x the size of the other three
    # c1, c2, c3, c4 = st.beta_columns((2, 1, 1, 1))
    # st.sidebar.header('Function')
    # original = Image.open('picture/cat.jpg')
    # c1.header("Original")
    # c1.image(original, use_column_width=True)
    # c2.header("Original")
    # c2.image(original, use_column_width=True)
    # c2.image(original, use_column_width=True)
    # c3.header("Original")
    # c3.image(original, use_column_width=True)
    # c3.image(original, use_column_width=True)
    # c4.header("Original")
    # c4.image(original, use_column_width=True)
    # c4.image(original, use_column_width=True)
    # c5, c6 = st.beta_columns(2)
    # c5.image(original, use_column_width=True)
    # c6.image(original, use_column_width=True)
    # st.image(original, use_column_width=True)

    # st.line_chart({"data": [1, 5, 2, 6, 2, 1]})

    # with st.beta_expander("See explanation"):
    #     st.write("""
    #         The chart above shows some numbers I picked for you.
    #         I rolled actual dice for these, so they're *guaranteed* to
    #         be random.
    #     """)
    #     st.image("https://static.streamlit.io/examples/dice.jpg")

    # st.write('hello')

    # with st.empty():
    #     for seconds in range(10):
    #         st.write(f"⏳ {seconds} seconds have passed")
    #         time.sleep(1)
    #     st.write("✔️ 1 minute over!")

    # st.write('world')

    # placeholder = st.empty()
    # time.sleep(3)
    # placeholder.text("hello")
    # time.sleep(3)
    # placeholder.line_chart({"data":[1, 5, 2, 6]})
    # time.sleep(3)
    # with placeholder.beta_container():
    #     st.write("This is one elemet")
    #     st.write("This is another")
    # time.sleep(3)
    # placeholder.empty()
    
    # my_expander = st.beta_expander('')
    # my_expander.write('Hello there!')
    # clicked = my_expander.button('Click me!')

    # my_expander = st.beta_expander('')
    # with my_expander:
    #     'Hello there!'
    #     clicked = st.button('Click me!')

    def my_widget(key):
        st.subheader('Hello there!')
        clicked = st.button("Click me " + key)
    # This works in the main area
    clicked = my_widget("first")
    # And within an expander
    my_expander = st.beta_expander("Expand", expanded=True)
    with my_expander:
        clicked = my_widget("second")
    # AND in st.sidebar!
    with st.sidebar:
        clicked = my_widget("third")

    # st.markdown(
    # """
    # <style>
    # .sidebar .sidebar-content {
    #     background-image: linear-gradient(#2e7bcf,#2e7bcf);
    #     color: white;
    # }
    # </style>
    # """,unsafe_allow_html=True,)

    color = st.select_slider('Select a color of the rainbow', options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    st.write('My favorite color is', color)

    


if __name__=='__main__':
    main()