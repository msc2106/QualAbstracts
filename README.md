----
QualAbstracts: Summarizing qualitative social science and humanities research
----

# Problem Statement
Fine-tune a Large Language Model to produce abstract-length summaries of academic research in the social sciences and humanities in the discursive style characteristic of these disciplines.

# Context
The proliferation of Large Language Models (LLMs) has energized efforts to automate complex discursive tasks. In addition to various interactive chatbots, there are many efforts to produce tools to assist in writing and research. Elicit, for example, promises to act like an “AI research assistant,” able to carry out the work of combing through research literature, pulling out key findings and methodological issues. However, the functionality is clearly geared towards certain kinds of research, especially in the harder sciences and medicine. Other domains, such as the more qualitative branches of the social sciences and humanities pose a distinct set of problems that existing tools are less well-suited for.

The goal of this project is to begin to tackle the distinctive problems of automating summarizing in these more humanistic discursive domains. The upshot or contribution of a piece of research in one of these disciplines is best understood as an addition to a conversation. To capture it, then, it is necessary to match the context of that conversation. It often cannot be reduced to information. This represents an example of a more general challenge for Large Language Models: the reproduction of domain-specific style.

# Scope, Criteria for Success, and Constraints
The scope of this project is limited to generating abstracts (300 to 500-word summaries) based on the full text of research outputs in the social sciences and humanities. Its goal is to reproduce the characteristic style of this genre of writing. To do this, an open-source LLM (such as StableLM, OpenLLaMA, Google’s T5, or MPT) will be fine-tuned on pairs of research works and abstracts.

One key constraint arises from the quality of the data: although academic publishers make articles from their journals available online, they are heavily restricted. The works that are available for legal scraping are much more miscellaneous.

A second constraint is how to evaluate the results. There are metrics for evaluating supervised summarization (such as ROUGE), but this is ultimately based on matching sequences of words. This is at best a very rough proxy for discursive similarity, in particular to similarity of style, which is the main interest here. Accordingly, I will have to rely on my domain-specific knowledge as a social scientist to judge evaluate the ultimate success on this more diffuse goal.

# Stakeholders, or, Who Is This For?
Who needs automated abstracts? Researchers are quite capable of writing them for their own articles, and the amount of time saved by automation is likely limited. However, a more promising potential use of this type of model would not be established researchers but instead students. A major barrier of entry to the humanities and social sciences in the English-speaking world, in particular for non-native English speakers, is the mastery of these fields’ distinctive style in talking about one’s research. It is not something that is formally taught but instead is expected to be picked up by emulation of examples. For such students, AI summarization cannot legitimately replace the need for them to learn this skill—if for no other reason than that most members of the field would object to this—but it could serve as a learning aid.

# Data Sources
The data for this project is drawn from [CORE](https://core.ac.uk/), a UK-based aggregator for open-access research. Records are submitted by a large number of data providers — often repositories maintained by academic institutions. It provides a queryable API and returns various data on each record, including abstracts and full text. Its full database is 3.5TB, but for the purposes of this project, it will be adequate for retrieve several thousand research articles from the API.

This was supplemented by subject-matter tags from [CrossRef](https://www.crossref.org/), 

# Model and training

Comparison models:
- ccdv/lsg-bart-base-16384-arxiv
- google/bigbird-pegasus-large-arxiv