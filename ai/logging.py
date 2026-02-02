from torch.utils.tensorboard import SummaryWriter
sumwriter = SummaryWriter()
def log(self,test_loss:float=None,test_steps:int=0,avgScore:float=None):
    """logs the test_loss avgScore to the tensorboard
    Args:
    :param test_loss (float) test loss
    :param avgScore (float) average score over test playouts
    :param content (str) content to add to the log file
    :param test_steps (int) number of test steps
    """
    if test_loss and avgScore:
        sumwriter.add_scalar("Loss/train",test_loss,test_steps)
        sumwriter.add_scalar("Mean Reward",avgScore,test_steps)
    
