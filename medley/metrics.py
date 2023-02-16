import torch

from .configs import device


def mean_average_precision(
    query_code,
    retrieval_code,
    query_targets,
    retrieval_targets,
    top_k=None,
):
    query_code, retrieval_code, query_targets, retrieval_targets = (
        tensor.float().to(device)
        for tensor in (query_code, retrieval_code, query_targets, retrieval_targets)
    )

    num_query = query_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (
            retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t()
        )

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:top_k]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean().item()
    mean_AP = mean_AP / num_query
    return mean_AP


def pr_curve(query_code, retrieval_code, query_targets, retrieval_targets):
    query_code, retrieval_code, query_targets, retrieval_targets = (
        tensor.float().to(device)
        for tensor in (query_code, retrieval_code, query_targets, retrieval_targets)
    )

    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query, num_bit + 1).to(device)
    R = torch.zeros(num_query, num_bit + 1).to(device)
    for i in range(num_query):
        gnd = (
            (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0)
            .float()
            .squeeze()
        )
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (
            hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(device)
        ).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask

    return P, R
