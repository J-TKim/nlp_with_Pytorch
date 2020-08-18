import sys, fileinput
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    buf = []
    
    for line in fileinput.input():
        if line.strip() != "":
            buf += [line.strip()]
            sentences = sent_tokenize(" ".join(buf))
            
            if len(sentences) > 1:
                buf = sentences[-1:]
                
                sys.stdout.write("\n".join(\sentences[:-1]) + "\n")
                
    sys.stdout.write(" ".join(buf) + "\n")