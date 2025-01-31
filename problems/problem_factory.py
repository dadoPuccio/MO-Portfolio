from problems.portfolio_problem import PortfolioProblem

class ProblemFactory:
    @staticmethod
    def get_problem(problem_type: str, **kwargs):
        
        if problem_type == 'Portfolio':

            return PortfolioProblem(kwargs['prob_path'], 
                                    kwargs['s'],
                                    kwargs['scaling_factors'],
                                    kwargs['sparsity_tol'],
                                    kwargs['objectives'],
                                    kwargs['financial_index'],
                                    kwargs['scalarization'],
                                    kwargs['beta_bounds'])
                                    
        else:
            raise NotImplementedError
