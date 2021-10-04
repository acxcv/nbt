import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContextModeler(nn.Module):
    def __init__(self):
        super(ContextModeler, self).__init__()

    def forward(self, prev_sysout, candidate_pair, user_utt):
        # cs: tensor -> candidate slot embedding
        # cv: tensor -> candidate value embedding
        cs = candidate_pair[0].squeeze(0)
        cv = candidate_pair[1].squeeze(0)

        metrics_out = {
            'mc': [],
            'mr': []
        }

        for element in prev_sysout:

            ts, tv, tq, metric = None, None, None, None

            # slot-value pair case with a list containing two tensors
            if isinstance(element, list):
                ts = element[0].squeeze(0)
                tv = element[1].squeeze(0)

                left_expr = torch.dot(cs, ts)
                right_expr = torch.dot(cv, tv)
                intermediate = torch.multiply(left_expr, right_expr)

                request_metric = torch.multiply(intermediate, user_utt)
                metric = request_metric
                metrics_out['mr'].append(metric)

            # slot case with one tensor
            elif isinstance(element, torch.Tensor):
                tq = element.squeeze(0)
                # TODO: PASS OUTPUT FROM CNN
                left_expr = torch.dot(cs, tq)
                right_expr = user_utt

                confirm_metric = torch.multiply(left_expr, right_expr)

                metric = confirm_metric
                metrics_out['mc'].append(metric)

            # this should not happen but who knows
            else:
                raise ValueError('ERROR in ContextModeling.forward. '
                      'Input is not a tensor or tuple. \n')

        return metrics_out
