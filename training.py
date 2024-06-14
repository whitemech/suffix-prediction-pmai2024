import torch
import torchmetrics
from evaluation import evaluate_accuracy_next_activity, logic_loss
if torch.cuda.is_available():
     device = 'cuda:0'
else:
    device = 'cpu'
from statistics import mean

def train(rnn, train_dataset, test_dataset, max_num_epochs, epsilon, deepdfa = None, prefix_len = 0, batch_size= 64):

    curr_temp = 0.5
    lambda_temp = 0.9999999999
    min_temp = 0.0001
    rnn = rnn.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=rnn.parameters(), lr=0.0005)
    acc_func = torchmetrics.Accuracy(task="multiclass", num_classes=train_dataset.size()[-1], top_k=1).to(device)
    old_loss = 1000

    ############################ TRAINING
    for epoch in range(max_num_epochs*2):
        current_index = 0
        train_acc_batches = []
        sup_loss_batches = []
        log_loss_batches = []
        while current_index <= train_dataset.size()[0]:
            initial = current_index
            final = min(current_index + batch_size, train_dataset.size()[0] + 1)
            current_index = final
            # print(batch.size())
            X = train_dataset[initial:final, :-1, :].to(device)
            # print("X size:", X.size())
            Y = train_dataset[initial:final, 1:, :]
            # print(Y.size())
            target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)
            # print(target.size())

            optim.zero_grad()
            #################################### supervised loss
            predictions, _ = rnn(X)
            predictions = predictions.reshape(-1, predictions.size()[-1])

            sup_loss = loss_func(predictions, target)
            sup_loss_batches.append(sup_loss.item())

            ##################################### logic loss
            if epoch > 500 and deepdfa != None:
                log_loss = logic_loss(rnn, deepdfa, X, prefix_len, curr_temp)
                log_loss_batches.append(log_loss.item())
                loss = 0.6*sup_loss + 0.4*log_loss
            else:
                loss = sup_loss

            loss.backward()
            optim.step()

            train_acc_batches.append(acc_func(predictions, target).item())

        train_acc = mean(train_acc_batches)
        sup_loss = mean(sup_loss_batches)
        #curr_temp = max(lambda_temp*curr_temp, min_temp)
        if curr_temp == min_temp:
            print("MINIMUM TEMPERATURE REACHED")
        test_acc = evaluate_accuracy_next_activity(rnn, test_dataset, acc_func)
        if epoch % 100 == 0:
            if deepdfa == None or epoch <= 500:
                print("Epoch {}:\tloss:{}\ttrain accuracy:{}\ttest accuracy:{}".format(epoch, sup_loss, train_acc, test_acc))
                loss= sup_loss
            else:
                log_loss = mean(log_loss_batches)
                print("Epoch {}:\tloss:{}\tlogic_loss:{}\ttrain accuracy:{}\ttest accuracy:{}".format(epoch, sup_loss, log_loss, train_acc, test_acc))
                loss = 0.6*sup_loss + 0.4*log_loss

        if loss < epsilon:
            return train_acc, test_acc

        if epoch > 500 and abs(loss - old_loss) < 0.00001:
            return train_acc, test_acc
        old_loss = loss

    return train_acc, test_acc

