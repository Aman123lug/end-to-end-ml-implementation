import sys
import logging
def error_handler(error, error_msg:sys):
    _,_,exce_error = error_msg.exc_info()
    #  fetching the error script filename
    file_name = exce_error.tb_frame.f_code.co_filename
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exce_error.tb_lineno, str(error)
    )
    
class CustomErrorHandler(Exception):
    def __init__(self,error, error_msg):
        super().__init__(error)
        self.error = error_handler(error, error_msg)
        
    def __str__(self):
        return self.error
    
if __name__ == "__main__":
    try:
        a = 0/1
    except Exception as e:
        logging.info("division by zero")
        raise CustomErrorHandler(e, sys)
    