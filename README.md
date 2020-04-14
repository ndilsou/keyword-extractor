# Automatic Keyword Extraction with Streamlit

A simple streamlit app showing a keyword extraction report on a corpus.

## Usage

1. in your command line, run `make run-app` to pull and run the latest docker image for the app.
2. Once in the app, you can submit a corpus to be processed by dropping the zip file containing your dataset in the file uploaded.
3. Select the extraction model to use. currently supported are TfIdf, TextRank and EmbedRank. You can also select how
   many keywords to extract from each document.

In the main section of the page, you'll be able to see a summary of the extraction. You can also select a larger full
fledge report that will show the keywords in context.

Loading the EmbedRank model may take some time on the first run as the universal sentence encoder needs to be fetched from
Tensorflow Hub. It should get cached on subsequent executions.

## TODO
- allow user to select model parameters from sidebar.
- add a default corpus.
- add a filter for documents in the full report.
- add full support for EmbedRank++
