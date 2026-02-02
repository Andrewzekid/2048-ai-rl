from torch.utils.tensorboard import SummaryWriter
sumwriter = SummaryWriter()
def log(mode:str,loss:float=None,steps:int=0,score:float=None):
    """logs the test_loss avgScore to the tensorboard
    Args:
    :param mode (str), test or training
    :param test_loss (float) test loss
    :param avgScore (float) average score over test playouts
    :param content (str) content to add to the log file
    :param test_steps (int) number of test steps
    """
    if mode=="test":
        sumwriter.add_scalar("Loss/test",loss,steps)
        sumwriter.add_scalar("Mean Reward",score,steps)
    elif mode == "train":
        sumwriter.add_scalar("Loss/train",loss,steps)
    
