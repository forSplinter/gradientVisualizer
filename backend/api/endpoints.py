from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import numpy as np
from core_build.function import f_register
from core_build.loss_function import l_register
from core_build.optimizer import GradientDescend


app = FastAPI
router = APIRouter()


@router.get("/functions")
def get_function():
    return {"available math_functions": f_register.list_f()}


@router.get("/loss_functions")
def get_loss():
    return {"availalbe loss_functions": l_register.list_lf()}


class OptimizationRequest(BaseModel):
    function_name: str
    loss_function: str
    learning_rate: float = 0.01
    iterations: int = 100


@router.post("/optimize")
def optimize(request: OptimizationRequest):
    math_function = f_register.get_f(request.function_name)
    loss_function = l_register.get_lf(request.loss_function)

    if math_function is None or loss_function is None:
        return {"error": "Function or loss function not found"}

    optimizer = GradientDescend(
        math_function, loss_function, request.learning_rate, request.iterations
    )
    opt_param, loss_hitory = optimizer.optimize()

    return {
        "optimized_params": opt_param,
        "loss_hitory": loss_hitory,
    }
