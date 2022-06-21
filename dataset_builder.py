from base64 import encode
from config import *
from libs import *
from utils import prettify_text


class CustomDataset(Dataset):
    intab = list("ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ")
    outtab = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d"

    def __init__(
        self,
        sentences: List[str],
        labels: List[int],
        max_len: int,
        tokenizer: transformers.PreTrainedTokenizer,
        vncorenlp_jar_path: str,
        unknown_label_index:int
    ):
        self.replaces_dict = dict(zip(self.intab, self.outtab))
        self.r = re.compile("|".join(self.intab))
        self.max_len = max_len
        self.labels = labels
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vncorenlp_wseg = VnCoreNLP(
            vncorenlp_jar_path,
            annotators="wseg",
            max_heap_size="-Xmx500m",
        )
        self.unknown_label_index = unknown_label_index

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        label, sent = self.labels[index], self.sentences[index]
        sent = prettify_text(sent)

        # Remove Vietnamese tone marks randomly
        if sent and bool(random.getrandbits(1)):
            sent = self.r.sub(lambda m: self.replaces_dict[m.group(0)], sent)

        if not sent:
            sent = "hàng hóa không xác định"
            label = self.unknown_label_index

         # Word segment with vncorenlp
        sent = " ".join(self.vncorenlp_wseg.tokenize(sent)[0])

        # Automatically add special tokens: <s> and </s> already
        encoded_sent = self.tokenizer.encode(sent)

        # Truncating
        encoded_sent = encoded_sent[: self.max_len]
        # Padding
        encoded_sent += [0] * (self.max_len - len(encoded_sent))
        # Masking
        mask = [int(token_id > 0) for token_id in encoded_sent]

        return torch.tensor(encoded_sent), torch.tensor(mask), torch.tensor(label)
