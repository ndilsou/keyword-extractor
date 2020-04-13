import pytest
import spacy

@pytest.fixture('session')
def nlp():
    return spacy.load('en_core_web_md')

@pytest.fixture('session')
def simple_doc(nlp):
    return nlp('''Let me begin by saying thanks to all you who've traveled, from far and wide, to brave the cold today.
We all made this journey for a reason. It's humbling, but in my heart I know you didn't come here just for me, you came here because you believe in what this country can be. In the face of war, you believe there can be peace. In the face of despair, you believe there can be hope. In the face of a politics that's shut you out, that's told you to settle, that's divided us for too long, you believe we can be one people, reaching for what's possible, building that more perfect union.
That's the journey we're on today. But let me tell you how I came to be here. As most of you know, I am not a native of this great state. I moved to Illinois over two decades ago. I was a young man then, just a year out of college; I knew no one in Chicago, was without money or family connections. But a group of churches had offered me a job as a community organizer for $13,000 a year. And I accepted the job, sight unseen, motivated then by a single, simple, powerful idea - that I might play a small part in building a better America.
My work took me to some of Chicago's poorest neighborhoods. I joined with pastors and lay-people to deal with communities that had been ravaged by plant closings. I saw that the problems people faced weren't simply local in nature - that the decision to close a steel mill was made by distant executives; that the lack of textbooks and computers in schools could be traced to the skewed priorities of politicians a thousand miles away; and that when a child turns to violence, there's a hole in his heart no government could ever fill.
''')
