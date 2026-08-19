class MaximumLikelihoodTarget {
public:

    using ReflectionList = std::vector<ReflectionLikelihood>;

    MaximumLikelihoodTarget(
        const Simple6MosaicityParameterisation& model,
        const Eigen::Matrix3d& A,
        const Eigen::Vector3d& s0,
        const std::vector<Eigen::Vector3d>& xyzcal_px,
        const std::vector<Eigen::Vector3d>& xyzobs_px,
        const std::vector<Eigen::Vector3d>& covariances,
        const std::vector<double>& intensities,
        const std::vector<Eigen::Vector3i>& miller_indices,
        const Panel& panel);

    void update();

    double mse() const;

    double log_likelihood() const;

    ParameterVector first_derivatives() const;

    FisherMatrix fisher_information() const;

    const ReflectionList& reflections() const {
        return data_;
    }

private:

    const Simple6MosaicityParameterisation& model_;

    ReflectionList data_;
};

MaximumLikelihoodTarget::MaximumLikelihoodTarget(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& s0,
    const std::vector<Eigen::Vector3d>& xyzcal_mm,
    const std::vector<Eigen::Vector3d>& xyzobs_mm,
    const std::vector<Eigen::Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const Panel& panel)
    :
    model_(model)
{
    const std::size_t n = miller_indices.size();

    /*assert xyzcal_mm.size() == n;
    DIALS_ASSERT(xyzobs_mm.size() == n);
    DIALS_ASSERT(covariances.size() == n);
    DIALS_ASSERT(intensities.size() == n);*/

    data_.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {

        // FIXME is mobs, s1 in mm?
        Vector3d s1cal = panel.get_lab_coord(xyzcal_mm[i][0], xyzcal_mm[i][1]);
        Vector3d s1obs = panel.get_lab_coord(xyzobs_mm[i][0], xyzobs_mm[i][1]);
        Eigen::Vector2d mobs = s1obs.head<2>();
        Eigen::Matrix2d sobs;
        sobs << covariances[i][0], covariances[i][2],covariances[i][2], covariances[i][1];

        data_.emplace_back(
            model_,
            A,
            s0,
            s1cal,
            miller_indices[i],
            intensities[i],
            mobs,
            sobs);
    }
}

void MaximumLikelihoodTarget::update()
{
    for (auto& r : data_) {
        r.update();
    }
}

double MaximumLikelihoodTarget::log_likelihood() const
{
    double lnL = 0.0;

    for (const auto& r : data_) {
        lnL += r.log_likelihood();
    }

    return lnL;
}

double MaximumLikelihoodTarget::mse() const
{
    double mse = 0.0;

    for (const auto& r : data_) {

        Eigen::Vector2d diff =
            r.mobs() -
            r.conditional().mean();

        mse += diff.squaredNorm();
    }

    return mse / static_cast<double>(data_.size());
}

ParameterVector MaximumLikelihoodTarget::first_derivatives() const
{
    ParameterVector dL =
        ParameterVector::Zero();

    for (const auto& r : data_) {
        dL += r.first_derivatives();
    }

    return dL;
}

FisherMatrix
MaximumLikelihoodTarget::fisher_information() const
{
    FisherMatrix I =
        FisherMatrix::Zero();

    for (const auto& r : data_) {
        I += r.fisher_information();
    }

    return I;
}



/*class MaximumLikelihoodTarget:
    def __init__(
        self, model, s0, sp_list, h_list, ctot_list, mobs_list, sobs_list, panel_ids
    ):
        # Check input
        assert len(h_list) == sp_list.shape[-1]
        assert len(h_list) == ctot_list.shape[-1]
        assert len(h_list) == mobs_list.shape[-1]
        assert len(h_list) == sobs_list.shape[-1]

        # Save the model
        self.model = model

        # Compute the change of basis for each reflection
        self.data = []
        for i in range(len(h_list)):
            self.data.append(
                ReflectionLikelihood(
                    model,
                    s0,
                    sp_list[:, i],
                    matrix.col(h_list[i]),
                    ctot_list[i],
                    mobs_list[:, i],
                    sobs_list[:, :, i],
                    panel_ids[i],
                )
            )



    def rmsd(self):
        """
        The RMSD in pixels

        """
        mse_x = 0.0
        mse_y = 0.0
        for i in range(len(self.data)):
            R = self.data[i].R_cctbx
            mbar = tuple(self.data[i].conditional.mean().flatten())
            xobs = tuple(self.data[i].mobs.flatten())
            norm_s0 = self.data[i].norm_s0
            rse_i = rse(R, mbar, xobs, norm_s0, self.model.experiment.detector)
            mse_x += rse_i[0]
            mse_y += rse_i[1]
        mse_x /= len(self.data)
        mse_y /= len(self.data)
        return np.sqrt(np.array([mse_x, mse_y]))
        */
