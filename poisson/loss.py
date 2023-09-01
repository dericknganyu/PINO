from utils import *





    # class Poisson(torch.nn.Module):
    #     "Custom Poisson PDE definition for PINO"

    #     def __init__(self, gradient_method: str = "hybrid"):
    #         super().__init__()
    #         self.gradient_method = str(gradient_method)

    #     def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #         # get inputs
    #         u = input_var["sol"]
    #         c = input_var["coeff"]
    #         dcdx = input_var["Kcoeff_y"]  # data is reversed
    #         dcdy = input_var["Kcoeff_x"]

    #         dxf = 1.0 / u.shape[-2]
    #         dyf = 1.0 / u.shape[-1]
    #         # Compute gradients based on method
    #         # Exact first order and FDM second order
    #         if self.gradient_method == "hybrid":
    #             dudx_exact = input_var["sol__x"]
    #             dudy_exact = input_var["sol__y"]
    #             dduddx_fdm = ddx(
    #                 u, dx=dxf, channel=0, dim=0, order=1, padding="replication"
    #             )
    #             dduddy_fdm = ddx(
    #                 u, dx=dyf, channel=0, dim=1, order=1, padding="replication"
    #             )
    #             # compute poisson equation
    #             poisson = (
    #                 1.0
    #                 + (dcdx * dudx_exact)
    #                 + (c * dduddx_fdm)
    #                 + (dcdy * dudy_exact)
    #                 + (c * dduddy_fdm)
    #             )
    #         # FDM gradients
    #         elif self.gradient_method == "fdm":
    #             dudx_fdm = dx(u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
    #             dudy_fdm = dx(u, dx=dyf, channel=0, dim=1, order=1, padding="replication")
    #             dduddx_fdm = ddx(
    #                 u, dx=dxf, channel=0, dim=0, order=1, padding="replication"
    #             )
    #             dduddy_fdm = ddx(
    #                 u, dx=dyf, channel=0, dim=1, order=1, padding="replication"
    #             )
    #             # compute poisson equation
    #             poisson = (
    #                 1.0
    #                 + (dcdx * dudx_fdm)
    #                 + (c * dduddx_fdm)
    #                 + (dcdy * dudy_fdm)
    #                 + (c * dduddy_fdm)
    #             )
    #         # Fourier derivative
    #         elif self.gradient_method == "fourier":
    #             dim_u_x = u.shape[2]
    #             dim_u_y = u.shape[3]
    #             u = F.pad(
    #                 u, (0, dim_u_y - 1, 0, dim_u_x - 1), mode="reflect"
    #             )  # Constant seems to give best results
    #             f_du, f_ddu = fourier_derivatives(u, [2.0, 2.0])
    #             dudx_fourier = f_du[:, 0:1, :dim_u_x, :dim_u_y]
    #             dudy_fourier = f_du[:, 1:2, :dim_u_x, :dim_u_y]
    #             dduddx_fourier = f_ddu[:, 0:1, :dim_u_x, :dim_u_y]
    #             dduddy_fourier = f_ddu[:, 1:2, :dim_u_x, :dim_u_y]
    #             # compute poisson equation
    #             poisson = (
    #                 1.0
    #                 + (dcdx * dudx_fourier)
    #                 + (c * dduddx_fourier)
    #                 + (dcdy * dudy_fourier)
    #                 + (c * dduddy_fourier)
    #             )
    #         else:
    #             raise ValueError(f"Derivative method {self.gradient_method} not supported.")

    #         # Zero outer boundary
    #         poisson = F.pad(poisson[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
    #         # Return poisson
    #         output_var = {
    #             "poisson": dxf * poisson,
    #         }  # weight boundary loss higher