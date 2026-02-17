import sys

def error_message(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_mess = "Error occured in python \nScript Name : [{0}] \nLine Number : [{1}]\nError Message : [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_mess

class CustomException(Exception):
    def __init__(self,error_mess,error_details:sys):
        super().__init__(error_mess)
        self.error_message = error_message(error=error_mess,error_details=error_details)
    def __str__(self):
        return self.error_message