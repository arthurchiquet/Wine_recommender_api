import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from google_your_wine.data import get_data, get_params
from google_your_wine.utils import lowercase, compute_distance, split_url, clean_reviews, remove_punctuation, lemma, sno, return_mapped_descriptor
from google_your_wine.input_prep import clean_input, transform
from PIL import Image
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
import webbrowser


bad_words, n_grams, dict_tfidf, word2vec, descriptors = get_params('all')

def make_phrase(reviews):
    phrase = ''
    for i in reviews:
        phrase += " ".join(i)+" "
    return phrase

# def make_url(selection):
#     url = f'https://www.wine-searcher.com/find/{split_url(selection)}'
#     html = '<a href={}> >>>> Online shop</a>'.format(url)
#     div = Div(text=html)
#     st.bokeh_chart(div)

def plot_word_cloud(phrase):
    wc = WordCloud(width = 800, height = 800,background_color="white", max_words=15).generate(phrase)
    # plt.figure(figsize = (8, 5), facecolor = None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    st.pyplot()

def return_wines_bottle(df, user_selection, n_bottles, options, act_wine):
    input_vector = df[df.title==user_selection].index
    distances = compute_distance(np.array(df.iloc[:,11:111]), np.array(df.iloc[:,11:111].iloc[input_vector]), 'euclidean')
    distances_index = distances.argsort()
    if options == []:
        output_df = df.iloc[distances_index].iloc[1:n_bottles+1,:]
    else:
        output_df = df.iloc[distances_index][df.iloc[distances_index]['country'].isin(options)].iloc[1:n_bottles+1,:]
    output_df['proximity'] = MinMaxScaler().fit_transform(pd.DataFrame(distances[output_df.index.values]))
    output_df['points' ] = (MinMaxScaler().fit_transform(pd.DataFrame(output_df['points']))*50 + 50).round(0)
    if n_bottles > 100:
        stats = n_bottles
    else:
        stats = 100
    return output_df.reset_index(drop=True), df.iloc[distances_index][['country', 'variety']].head(stats)

def return_wines_flavors(df, input_vector, n_bottles, options):
    distances = compute_distance(np.array(df.iloc[:,11:111]), input_vector, 'euclidean')
    distances_index = distances.argsort()
    if options == []:
        output_df = df.iloc[distances_index].head(n_bottles)
    else:
        provinces_list = df[df.country.isin(options)].province.unique()
        provinces = st.sidebar.multiselect('Choose your target provinces:',provinces_list)
        if provinces == []:
            output_df = df.iloc[distances_index][df.iloc[distances_index]['country'].isin(options)].head(n_bottles)
        else:
            output_df = df.iloc[distances_index][(df.iloc[distances_index]['country'].isin(options)) & (df.iloc[distances_index]['province'].isin(provinces))].head(n_bottles)
    output_df['proximity'] = MinMaxScaler().fit_transform(pd.DataFrame(distances[output_df.index.values]))
    output_df['points' ] = (MinMaxScaler().fit_transform(pd.DataFrame(output_df['points']))*50 + 50).round(0)
    if n_bottles > 100:
        stats = n_bottles
    else:
        stats = 100
    return output_df.reset_index(drop=True), df.iloc[distances_index][['country', 'variety']].head(stats)

def key_words(text):
    text = remove_punctuation(text)
    text = text.lower()
    text = ''.join(word for word in text if not word.isdigit())
    text_token = text.split()
    clean_text = [word for word in text_token if not word in bad_words]
    clean_text = lemma(clean_text)
    clean_text = sno(clean_text)
    clean_text = n_grams[clean_text]
    clean_text = [return_mapped_descriptor(descriptors, word) for word in clean_text]
    return clean_text

def bottle_mode(user_input, df, get_params):
    if user_input != "":
        user_selection = st.sidebar.selectbox('Select a wine in the database:', df[((pd.concat([df.title.map(lowercase).str.contains(word) for word in user_input.lower().split()], axis=1)).sum(1) > len(user_input.lower().split())-1)].title.tolist())
        n_bottles = st.sidebar.number_input(label='Type the number of closest bottles:', value=10)
        if n_bottles > 100:
            stats = n_bottles
        else:
            stats = 100
        options = st.sidebar.multiselect('Choose your target countries:',np.sort(df.country.unique()))
        if user_selection == None:
            error_img = Image.open('pictures/error404.png')
            st.image(error_img, use_column_width=True)
        else :
            act_wine = df[df.title==user_selection].reset_index().drop(['index'], axis=1)
            output_df, country_var = return_wines_bottle(df, user_selection, n_bottles, options, act_wine)
            output_reviews = make_phrase([clean_reviews(review, bad_words, n_grams, descriptors) for review in output_df.description])
            output_df['key_words'] = output_df['description'].map(key_words)

            desc = act_wine.iloc[0].description
            point = act_wine.iloc[0].points
            price = act_wine.iloc[0].price
            act_country = act_wine.iloc[0].country

            st.sidebar.markdown("<p style='text-align: center; font-style: italic; color: Gray;'>Made with &#10084;&#65039;@LeWagon in Paris by Arthur, Brahim, Valentin & Nico</p>", unsafe_allow_html=True)

            banner = Image.open('pictures/banner.png')
            st.image(banner, use_column_width=True)

            st.markdown(f"<p style='font-size: 30px; font-weight: bold'> Details of Your Favorite Bottle: </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 16px; font-weight: bold'> {user_selection} </p>", unsafe_allow_html=True)

            st.dataframe(data=act_wine[['country', 'province', 'winery', 'variety', 'vintage']], height=600)
            st.markdown(f"<p style='font-size: 16px; font-style: italic'> '{desc}' </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 16px; text-align: center;'> Note: <span style='color: #34a05e; font-weight: bold'>{point}</span> / 100 <span style='padding:20px'></span> Price: <span style='color: #34a05e; font-weight: bold'>{price}</span> €</p>", unsafe_allow_html=True)
            wordcloud = st.checkbox('Word Cloud')
            if wordcloud:
                st.markdown(f"<p style='font-size: 20px; font-weight: bold'> Main aromas of your recommended wines: </p>", unsafe_allow_html=True)
                plot_word_cloud(output_reviews)

            st.markdown(f"<p style='font-size: 30px; font-weight: bold; padding-top: 50px;'>Mapping of <span style='color: #34a05e; font-weight: bold'>{n_bottles}</span> Most Comparable Bottles:</compara></p>", unsafe_allow_html=True)
            st.markdown(f"Based on reviews of Wine Magazines and our revolutionary NLP algorithm, we have identified the <span style='color: #34a05e; font-weight: bold'>{n_bottles}</span> bottles with the closest characteristics compared to the wine you selected. They are plotted below as per their flavor proximity (<em>X</em>), price (<em>Y</em>), country (<em>Color</em>) and rating (<em>Size</em>):", unsafe_allow_html=True)

            chart = alt.Chart(output_df).mark_point().encode(x='proximity', y='price', size='points', color='country', tooltip=['title', 'price', 'points', 'country', 'province', 'variety', 'vintage'])
            chart = chart.encode(alt.Y('price', scale=alt.Scale(domain=[output_df.price.min(),output_df.price.max()])))
            chart = chart.encode(alt.X('proximity', scale = alt.Scale(domain = [output_df.proximity.min(),output_df.proximity.max()])))
            st.altair_chart(chart, use_container_width=True)
            st.markdown(f"Main stats of the <span style='color: #34a05e; font-weight: bold'>{stats}</span> bottles with the closest characteristics compared to the wine you selected.", unsafe_allow_html=True)
            st.altair_chart(alt.Chart(country_var).mark_bar().encode(x='variety',y = 'count(variety)',color ='country'), use_container_width = True)

            for i in range (n_bottles):
                st.markdown(f"<p style='font-size: 15px; font-weight: bold;'>{i+1} : {output_df.title.iloc[i]}", unsafe_allow_html=True)
                st.markdown(f'" {output_df.description.iloc[i]} "', unsafe_allow_html=True)
                # url = f'https://www.wine-searcher.com/find/{split_url(output_df.title.iloc[i])}'
                # if st.button(f"Buy {i+1}"):
                #     webbrowser.open_new_tab(url)

            # selection = st.selectbox('Choose your wine:', output_df.title.tolist())
            # make_url(selection)

            st.markdown("<p style='text-align: center; padding-top: 50px; font-weight: bold;'>&#127863; Drink Cautiously &#127863;</p>", unsafe_allow_html=True)


def flavors_mode(user_input, df, get_params):
    if user_input != "":
        n_bottles = st.sidebar.number_input(label='Type the number of closest bottles:', value=10)
        if n_bottles > 100:
            stats = n_bottles
        else:
            stats = 100
        options = st.sidebar.multiselect('Choose your target countries:',np.sort(df.country.unique()))

        input_clean = clean_input(user_input, bad_words, n_grams)
        input_vector = transform(input_clean, word2vec, dict_tfidf)

        output_df, country_var = return_wines_flavors(df, input_vector, n_bottles, options)
        output_reviews = make_phrase([clean_reviews(review, bad_words, n_grams, descriptors) for review in output_df.description])
        output_df['key_words'] = output_df['description'].map(key_words)

        st.sidebar.markdown("<p style='text-align: center; font-style: italic; color: Gray;'>Made with &#10084;&#65039;@LeWagon in Paris by Arthur, Brahim, Valentin & Nico</p>", unsafe_allow_html=True)

        banner = Image.open('pictures/banner.png')
        st.image(banner, use_column_width=True)

        wordcloud = st.checkbox('Word Cloud')
        if wordcloud:
            st.markdown(f"<p style='font-size: 20px; font-weight: bold'> Main aromas of your recommended wines: </p>", unsafe_allow_html=True)
            plot_word_cloud(output_reviews)

        st.markdown(f"<p style='font-size: 30px; font-weight: bold; padding-top: 50px;'>Mapping of <span style='color: #34a05e; font-weight: bold'>{n_bottles}</span> Most Comparable Bottles:</compara></p>", unsafe_allow_html=True)
        st.markdown(f"Based on reviews of Wine Magazines and our revolutionary NLP algorithm, we have identified the <span style='color: #34a05e; font-weight: bold'>{n_bottles}</span> bottles with the closest characteristics compared to your description. They are plotted below as per their flavor proximity (<em>X</em>), price (<em>Y</em>), country (<em>Color</em>) and rating (<em>Size</em>):", unsafe_allow_html=True)

        chart = alt.Chart(output_df).mark_point().encode(x='price', y='points', color='country', size='points', tooltip=['title', 'price', 'points', 'country', 'province', 'variety', 'vintage'])
        chart = chart.encode(alt.Y('price', scale=alt.Scale(domain=[output_df.price.min(),output_df.price.max()])))
        chart = chart.encode(alt.X('proximity', scale = alt.Scale(domain = [output_df.proximity.min(),output_df.proximity.max()])))
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f"Main stats of the <span style='color: #34a05e; font-weight: bold'>{stats}</span> bottles with the closest characteristics compared to the wine you selected.", unsafe_allow_html=True)
        st.altair_chart(alt.Chart(country_var).mark_bar().encode(x='variety',y = 'count(variety)',color ='country'), use_container_width = True)

        for i in range (n_bottles):
            st.markdown(f"<p style='font-size: 15px; font-weight: bold;'>{i+1} : {output_df.title.iloc[i]}", unsafe_allow_html=True)
            st.markdown(f'" {output_df.description.iloc[i]} "', unsafe_allow_html=True)

        # selection = st.selectbox('Choose your wine:', output_df.title.tolist())
        # make_url(selection)

        st.markdown("<p style='text-align: center; padding-top: 50px; font-weight: bold;'>&#127863; Drink Cautiously &#127863;</p>", unsafe_allow_html=True)

