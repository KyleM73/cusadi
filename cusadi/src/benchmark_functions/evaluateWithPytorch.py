import torch
import casadi as ca

def evaluateWithPytorch(output, work_tensor, input_batch,
                        operations, output_idx, input_idx, const_instr,
                        num_instructions):
    for k in range(num_instructions):
        op = operations[k]
        o = output_idx[k]
        i = input_idx[k]
        if(op==ca.OP_CONST):
            work_tensor[:, o[0]] = const_instr[k]
        else:
            if op==ca.OP_INPUT:
                work_tensor[:, o[0]] = input_batch[i[0]][:, i[1]]
            elif op==ca.OP_OUTPUT:
                output[o[0]][:, o[1]] = work_tensor[:, i[0]]
            elif op==ca.OP_ADD:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] + work_tensor[:, i[1]]
            elif op==ca.OP_SUB:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] - work_tensor[:, i[1]]
            elif op==ca.OP_NEG:
                work_tensor[:, o[0]] = -work_tensor[:, i[0]]
            elif op==ca.OP_MUL:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] * work_tensor[:, i[1]]
            elif op==ca.OP_DIV:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] / work_tensor[:, i[1]]
            elif op==ca.OP_SIN:
                work_tensor[:, o[0]] = torch.sin(work_tensor[:, i[0]])
            elif op==ca.OP_COS:
                work_tensor[:, o[0]] = torch.cos(work_tensor[:, i[0]])
            elif op==ca.OP_TAN:
                work_tensor[:, o[0]] = torch.tan(work_tensor[:, i[0]])
            elif op==ca.OP_SQ:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] * work_tensor[:, i[0]]
            elif op==ca.OP_SQRT:
                work_tensor[:, o[0]] = torch.sqrt(work_tensor[:, i[0]])
            else:
                raise Exception('Unknown CasADi operation: ' + str(op))
    return output