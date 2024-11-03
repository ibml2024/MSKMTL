class resnet34WithFC(nn.Module):
    def __init__(self, num_classes=128):
        super(resnet34WithFC, self).__init__()
        pretrained_model = torchvision.models.resnet34(weights='DEFAULT')
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        num_freeze_layers = 17
        layer_count = 0
        for name, param in self.features.named_parameters():
            if layer_count < num_freeze_layers:
                param.requires_grad = False
            else:
                break
            layer_count += 1
        self.proj = nn.Linear(512, num_classes)
        self.proj.weight.data.normal_(0, 0.02)
        self.proj.bias.data.zero_()
        

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=0)
        return x
        
        
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BertModel.from_pretrained('./bert-base')
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(768, 128)
        self.proj.weight.data.normal_(0, 0.02)
        self.proj.bias.data.zero_()
            

    def forward(self, x: torch.Tensor):
        outputs = self.text_model(x)
        cls_feature = outputs.last_hidden_state[:, 0, :]
        x = self.proj(cls_feature)
        x = F.normalize(x, p=2, dim=1)
        return  x
        
        
        


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class LSTMmodel(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim=128, embed_dim=128, decoder_dim=128, vocab_size=256, encoder_dim=128, dropout=0.5, device = None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(LSTMmodel, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        
        self.device = device

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        #mean_encoder_out =  = encoder_out.mean(dim=1)
        mean_encoder_out = torch.squeeze(encoder_out,1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths,device):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)


        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1

        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    



    def generate1(self, encoder_out, beam_size=3, max_decoding_length=64, penalty_factor=0.3):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)
        start_token = 1
        end_token = 2

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        # Start decoding
        h, c = self.init_hidden_state(encoder_out)
        k_prev_words = torch.LongTensor([[start_token]]).to(self.device)

        # Initialize the beam search
        sequences = [[list(), 1.0, h, c, 0]]  # list of [sequence, score, hidden_state, cell_state, cummulative_penalty]

        # Start beam search
        for _ in range(max_decoding_length):
            all_candidates = []
            for i in range(len(sequences)):
                seq, score, h, c, cum_penalty = sequences[i]
                k_prev_word = seq[-1] if seq else start_token
                k_prev_word = torch.LongTensor([k_prev_word]).to(self.device)
     
                embedded_words = self.embedding(k_prev_word).squeeze(1)
                embedded_words = embedded_words.unsqueeze(0)
                h_expanded = h.unsqueeze(1).expand_as(embedded_words)
                
                input_to_decoder = torch.cat([embedded_words, h_expanded], dim=2)  # Concatenate embedded_words and h_expanded
                input_to_decoder = input_to_decoder.view(input_to_decoder.size(0), -1)  # Flatten to 2D tensor
                
                h, c = self.decode_step(input_to_decoder, (h, c))
                logits = self.fc(h)
                values, indices = torch.topk(logits, beam_size)
                topk_probs = F.softmax(values, 1)

                # Expand each current candidate
                for j in range(beam_size):
                    candidate = [seq + [indices[0][j].item()], score * topk_probs[0][j].item(), h, c, cum_penalty]

                    # Penalty for repeated words
                    if indices[0][j].item() in seq:
                        candidate[-1] += penalty_factor

                    all_candidates.append(candidate)

            # Order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

            # Select topk
            sequences = ordered[:beam_size]

        final_sequence = max(sequences, key=lambda tup: tup[1])
        return final_sequence[0]
    
    def generate(self, encoder_out, beam_size=3, max_decoding_length=64, penalty_factor=0.3):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)
        start_token = 1
        end_token = 2

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        # Start decoding
        h, c = self.init_hidden_state(encoder_out)
        k_prev_words = torch.LongTensor([[start_token]]).to(self.device)

        # Initialize the beam search
        sequences = [[list(), 1.0, h, c, 0]]  # list of [sequence, score, hidden_state, cell_state, cummulative_penalty]

        # Start beam search
        for _ in range(max_decoding_length):
            all_candidates = []
            for i in range(len(sequences)):
                seq, score, h, c, cum_penalty = sequences[i]
                k_prev_word = seq[-1] if seq else start_token
                k_prev_word = torch.LongTensor([k_prev_word]).to(self.device)

                embedded_words = self.embedding(k_prev_word).squeeze(1)
                embedded_words = embedded_words.unsqueeze(0)  # Add batch dimension

                h_expanded = h.unsqueeze(1).expand_as(embedded_words)  # Adjust h's dimension to match embedded_words

                input_to_decoder = torch.cat([embedded_words, h_expanded], dim=2)  # Concatenate embedded_words and h_expanded
                input_to_decoder = input_to_decoder.view(input_to_decoder.size(0), -1)  # Flatten to 2D tensor
                h, c = self.decode_step(input_to_decoder, (h, c))
                logits = self.fc(h)
                probabilities = F.softmax(logits, 1)

                # Randomly sample one word index from the predicted probabilities
                sampled_word_index = torch.multinomial(probabilities, 1).squeeze().item()

                # Expand each current candidate
                for j in range(beam_size):
                    candidate = [seq + [sampled_word_index], score * probabilities[0][sampled_word_index].item(), h, c, cum_penalty]

                    # Penalty for repeated words
                    if sampled_word_index in seq:
                        candidate[-1] += penalty_factor

                    all_candidates.append(candidate)

            # Order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

            # Select topk
            sequences = ordered[:beam_size]

        final_sequence = max(sequences, key=lambda tup: tup[1])
        return final_sequence[0]
        
        
        
        
class Memory(nn.Module):
    def __init__(self, radius=1, n_slot=4, in_dim=128, dim=128):
        super().__init__()
        self.radius = radius
        self.slot = n_slot
        self.dim = dim
        self.in_dim = in_dim
        
        self.value = nn.Parameter(torch.Tensor(n_slot, dim), requires_grad=True)
        nn.init.normal_(self.value, 0, 0.1)

        self.linear = nn.Linear(dim, in_dim)
        self.linear.weight.data.normal_(0, 0.02)
        self.linear.bias.data.zero_()
        self.q_embd = nn.Linear(128, dim)
        self.q_embd.weight.data.normal_(0, 0.02)
        self.q_embd.bias.data.zero_()
        
        self.softmax2 = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, query, target=None, inference=False):
        recon_loss, contrastive_loss = torch.zeros(1), torch.zeros(1)

        embd_query = self.q_embd(query)
        
        key_sim = torch.einsum('md,hsd->hsm', self.value, embd_query)
        key_add = self.softmax2(key_sim)
        
        m_head = torch.matmul(key_add, self.value.detach())

        if not inference:
            value_norm = F.normalize(self.value, dim=1)
            contrastive_loss = torch.abs(torch.eye(self.slot).to(query.device) - torch.matmul(value_norm, value_norm.transpose(0, 1))).sum()
            contrastive_loss = contrastive_loss.unsqueeze(0)
                
            recon_loss = torch.abs(1.0 - F.cosine_similarity(target, m_head.detach(), dim=-1))
            recon_loss = recon_loss.mean(0)

        return m_head, contrastive_loss, recon_loss
        
        
        
class MSKMTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.imgEncoder = resnet34WithFC()
        self.textEncoder = TextEncoder()
        self.lstm = LSTMmodel(device  = device)

        self.kg = Memory(radius=1, n_slot=32, in_dim=32, dim=128)
        self.fc1 = nn.Linear(256, 1)
        self.fc1.weight.data.normal_(0, 0.02)
        self.fc1.bias.data.zero_()

        self.dropout = nn.Dropout(p=0.1)
    
    
    def forward(self, image, ids, cli, ids_new, l, current_state):
        image_features = self.imgEncoder(image) 
        text_features = self.textEncoder(ids)
        df,contrastive_loss,recon_loss = self.kg(image_features.unsqueeze(1), text_features)

        x = self.fc1(torch.cat((image_features.unsqueeze(1), df), dim=-1))
        x = torch.tanh(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)

        lstmOUT = self.lstm(torch.squeeze(df, 0),ids_new,l,device)
        
        return lstmOUT, image_features, text_features, x,contrastive_loss,recon_loss